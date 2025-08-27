#!/usr/bin/env python3
"""
Fixed model_backend.py
- Uses the same HybridCNNViT definition you used in Colab
- Robust checkpoint loading (strict -> remap -> non-strict)
- Safe Grad-CAM++ (hooks last Conv2d)
- Keeps PDF/report generation code
"""

import os
import re
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.metrics import jaccard_score
from torchvision import transforms, models
from PIL import Image


# ------------------------------
# CONFIG
# ------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "/Users/keerthevasan/Documents/Study/Tumor_frontend/hybrid_cnn_vit_best.pt"  # update if needed

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Model (same as your Colab definition)
# ------------------------------
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, : x.size(1), :]

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, d_model=256, nhead=8, dim_ff=512, num_layers=4, drop=0.1):
        super().__init__()
        # ResNet18 backbone
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            # fallback for older torchvision versions
            resnet = models.resnet18(pretrained=True)
        self.cnn_stem = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, Hf, Wf]

        # project 512 -> d_model
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        # default seq_len = 7x7 = 49 for 224x224 input
        self.seq_len = 49
        self.pos = LearnablePositionalEncoding(self.seq_len, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=drop, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        feat = self.cnn_stem(x)            # [B, 512, 7, 7]
        feat = self.proj(feat)             # [B, d_model, 7, 7]
        B, C, Hf, Wf = feat.shape
        seq = feat.flatten(2).permute(0, 2, 1)  # [B, 49, d_model]

        # if seq_len differs from pos encoding, reinit (keeps parameters consistent across runs)
        if seq.size(1) != self.pos.pos.size(1):
            # Recreate positional encoding with correct length (move to same device later)
            self.pos = LearnablePositionalEncoding(seq.size(1), seq.size(2))

        seq = self.pos(seq)                # add positional encoding
        z = self.transformer(seq)          # [B, N, d_model]
        z = z.mean(dim=1)                  # global average pool
        z = self.norm(z)
        logits = self.head(z)
        return logits, feat

# ------------------------------
# Optional remapper (handles common name differences)
# ------------------------------
def remap_checkpoint_keys(sd_in):
    """
    Try to rename keys from older checkpoint naming to this model's names.
    This handles common differences like 'cnn_stem' vs 'cnn_features' etc.
    It's conservative: keys not matching will be carried through unchanged.
    """
    sd_out = {}
    for k, v in sd_in.items():
        new_k = k

        # map cnn_stem -> cnn_stem (if older checkpoint used cnn_stem it's fine)
        # map possible 'cnn_features' to 'cnn_stem' (some earlier code used that)
        if new_k.startswith("cnn_features."):
            new_k = new_k.replace("cnn_features.", "cnn_stem.")
        if new_k.startswith("cnn_stem."):
            # keep as-is
            pass

        # map projection name differences (proj -> proj)
        if new_k.startswith("proj."):
            new_k = new_k.replace("proj.", "proj.")

        # map positional enc key variants to 'pos.pos' -> 'pos.pos' (our pos is nested)
        # some checkpoints may name it 'pos' or 'vit_pos_embed'
        if new_k == "vit_pos_embed" or new_k == "pos.pos" or new_k == "pos":
            new_k = "pos.pos"  # our positional param is stored at self.pos.pos in earlier remap attempts
            # but our current model expects self.pos.pos parameter via LearnablePositionalEncoding.pos
            # We'll also attempt to load into 'pos.pos' or 'pos' afterwards; keep both keys
            sd_out[new_k] = v
            continue

        # transformer layers: if old checkpoint used "transformer.layers.i." map to "transformer.layers.i."
        # (we keep names as-is; transformer is nn.TransformerEncoder with submodule naming consistent)
        # final head mapping: 'head' might be 'head' or 'fc_out' — here model uses 'head'
        if new_k.startswith("head."):
            new_k = new_k  # keep 'head' as this model's head is named 'head'

        sd_out[new_k] = v
    return sd_out

# ------------------------------
# Helper: find last Conv2d in cnn_stem (for Grad-CAM hook)
# ------------------------------
def find_last_conv_module(module: nn.Module):
    last_conv = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

# ------------------------------
# Load checkpoint (strict -> remap -> non-strict)
# ------------------------------
def load_checkpoint_into_model(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
    state_dict_in = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    print(f"[Info] Incoming keys: {len(state_dict_in)}")

    # First try strict load (best)
    try:
        model.load_state_dict(state_dict_in, strict=True)
        print("[Info] Loaded checkpoint with strict=True (exact match).")
        return
    except Exception as e:
        print("[Info] strict=True failed:", str(e).splitlines()[0])

    # Try remapping keys and non-strict load
    state_dict_mapped = remap_checkpoint_keys(state_dict_in)
    print(f"[Info] Mapped keys: {len(state_dict_mapped)} (attempting non-strict load)")

    missing, unexpected = model.load_state_dict(state_dict_mapped, strict=False)
    print("Missing keys after non-strict load:", missing)
    print("Unexpected keys after non-strict load:", unexpected)
    # If positional embedding shape mismatch remains, try to adapt:
    # e.g. checkpoint pos [1,49,256] vs model pos [1,49,256] is fine; otherwise we skip checkpoint pos
    # (we already printed missing/unexpected so you can debug further)

# ------------------------------
# Grad-CAM++ implementation
# ------------------------------
class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_f = self.target_layer.register_forward_hook(self._fwd)
        # register_full_backward_hook is available modern PyTorch; fallback handled by try/except
        try:
            self.hook_b = self.target_layer.register_full_backward_hook(self._bwd)
        except Exception:
            self.hook_b = self.target_layer.register_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        try:
            self.hook_f.remove()
        except Exception:
            pass
        try:
            self.hook_b.remove()
        except Exception:
            pass

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        logits, _ = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.tensor(eps).to(alpha_denom.device))
        alphas = alpha_num / alpha_denom
        weights = torch.sum(alphas * F.relu(grads), dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1)
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ------------------------------
# Run Grad-CAM++ and produce results
# ------------------------------
def run_gradcampp(model, img_path, mask_path=None, threshold=0.3, out_prefix="gradcam_"):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs, _ = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    pred_class = classes[predicted.item()]
    conf_score = float(confidence.item() * 100.0)

    # Find last conv2d in cnn_stem to hook
    target_layer = find_last_conv_module(model.cnn_stem)
    if target_layer is None:
        # fallback to hooking entire stem if needed (less ideal)
        target_layer = model.cnn_stem

    gcampp = GradCAMpp(model, target_layer)
    cam = gcampp(input_tensor, class_idx=predicted.item())
    gcampp.remove()

    cam_resized = cv2.resize(cam, (224, 224))
    cam_bin = (cam_resized > threshold).astype(np.uint8)


    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224,224))
        mask = (mask > 127).astype(np.uint8)
        iou = float(jaccard_score(mask.flatten(), cam_bin.flatten()))

    orig = np.array(img.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
    gradcam_filename = f"{out_prefix}{os.path.basename(img_path)}"
    cv2.imwrite(gradcam_filename, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ---- Tumor ROI from Grad-CAM activations ----
    # ---- Tumor ROI from Grad-CAM activations ----
    roi_filename = None

    thr = 0.7  # try 0.5–0.7 depending on how tight you want
    cam_mask = (cam_resized >= thr).astype(np.uint8) * 255

    contours, _ = cv2.findContours(cam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        roi_img = np.array(img.resize((224,224))).copy()
        cv2.rectangle(roi_img, (x,y), (x+w, y+h), (0,255,0), 2)  # green box

        # --- Confidence score ---
        conf_value = float(confidence)   # convert tensor → float
        conf_text = f"{conf_value*100:.1f}%"

        # place text at top-right corner of green square (attached)
        text_scale = 0.4   # small font
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(conf_text,
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    text_scale,
                                                    text_thickness)

        # coordinates: attach to top-right corner of the rectangle
        text_x = x + w - text_w - 2   # inside the green box with padding
        text_y = y + text_h + 2       # just below top edge

        # optional background for readability
        cv2.rectangle(roi_img,
                    (text_x-1, text_y-text_h-2),
                    (text_x+text_w+1, text_y+baseline+2),
                    (0,255,0), -1)

        cv2.putText(roi_img,
                    conf_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (0,0,0),  # black text for contrast
                    text_thickness,
                    cv2.LINE_AA)

        roi_filename = f"{out_prefix}roi_{os.path.basename(img_path)}"
        cv2.imwrite(roi_filename, cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))




    results = {
        "pred_class": pred_class,
        "confidence": round(conf_score, 2),
        "gradcam_file": gradcam_filename,
        "roi_file":roi_filename
    }
    return results


# ------------------------------
# MAIN: load, test, gradcam, report
# ------------------------------
if __name__ == "__main__":
    # 1) Build model and load weights
    model = HybridCNNViT(num_classes=len(classes)).to(DEVICE)
    load_checkpoint_into_model(model, MODEL_PATH)
    model.eval()

    # 2) Sanity forward pass & Grad-CAM on a sample image
    img_path = "/Users/keerthevasan/Documents/Study/tumor/Data/Tumor/glioma_tumor/G_149.jpg"  # change if needed
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Sample image not found: {img_path}")

    results = run_gradcampp(model, img_path, mask_path=None, threshold=0.3)
    print("Prediction Results:", results)

    # 3) Create LLM-generated text (if ollama available)
  
