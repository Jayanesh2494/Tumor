

# ---------------------------
# Imports
# ---------------------------
import os, io
import cv2
import ollama
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, send_from_directory
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import timm

# ---------------------------
# CONFIG
# ---------------------------
CONFIG = {
    'model_paths': {
        'ensemble': '/Users/keerthevasan/Downloads/brain_tumor_classification_project/models/ensemble_brain_tumor.pth'
    },
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'image_size': 224
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['RESULT_FOLDER'] = "results"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ---------------------------
# Ensemble Model
# ---------------------------
class EnsembleModel(nn.Module):
    def __init__(self, efficientnet_model, vit_model, num_classes=4):
        super().__init__()
        self.efficientnet = efficientnet_model
        self.vit = vit_model
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        self.attention = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        eff_logits = self.efficientnet(x)
        vit_logits = self.vit(x)
        eff_probs = F.softmax(eff_logits, dim=1)
        vit_probs = F.softmax(vit_logits, dim=1)
        combined_probs = torch.cat([eff_probs, vit_probs], dim=1)
        attention_weights = self.attention(combined_probs)
        weighted_eff = attention_weights[:, 0:1] * eff_probs
        weighted_vit = attention_weights[:, 1:2] * vit_probs
        weighted_combined = weighted_eff + weighted_vit
        fusion_output = self.fusion(combined_probs)
        final_output = 0.7 * fusion_output + 0.3 * weighted_combined
        return final_output

class EfficientNetModelFixed(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=False)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, num_classes)
    def forward(self, x): return self.backbone(x)

class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        n_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(n_features, num_classes)
    def forward(self, x): return self.backbone(x)

def load_ensemble_model():
    eff = EfficientNetModelFixed(num_classes=4)
    vit = VisionTransformerModel(num_classes=4)
    ensemble = EnsembleModel(eff, vit, num_classes=4)

    checkpoint = torch.load(CONFIG['model_paths']['ensemble'], map_location=CONFIG['device'])
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        ensemble.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        ensemble.load_state_dict(checkpoint, strict=False)

    ensemble = ensemble.to(CONFIG['device'])
    ensemble.eval()
    return ensemble

def get_image_transform():
    return transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ---------------------------
# GradCAM++
# ---------------------------
class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients, self.activations, self.hook_handles = None, None, []
        self.target_layer = self._find_target_layer()
        if self.target_layer: self._register_hooks()

    def _find_target_layer(self):
        try:
            eff = self.model.efficientnet.backbone
            return eff.conv_head if hasattr(eff, "conv_head") else eff.blocks[-1]
        except: return None

    def _register_hooks(self):
        def fwd_hook(m, i, o): self.activations = o.detach()
        def bwd_hook(m, gi, go): self.gradients = go[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(bwd_hook))

    def generate_cam(self, input_image, class_idx):
        self.model.eval()
        output = self.model(input_image)
        target_score = output[0, class_idx]
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        gradients, activations = self.gradients, self.activations
        if gradients is None or activations is None: return None
        b, k, h, w = gradients.size()
        alpha_num = gradients.pow(2)
        alpha_denom = (2*gradients.pow(2) + activations.mul(gradients.pow(3)).view(b,k,h*w).sum(-1,keepdim=True).view(b,k,1,1))
        alpha = alpha_num / (alpha_denom + 1e-7)
        weights = (alpha * F.relu(gradients)).view(b,k,h*w).sum(-1).view(b,k,1,1)
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224,224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu()
        return ((cam - cam.min())/(cam.max()-cam.min())).numpy()

    def cleanup(self):
        for h in self.hook_handles: h.remove()
        self.hook_handles.clear()

def build_gradcam_overlay_pil(original_pil, cam):
    img_resized = np.array(original_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


import re

def clean_ai_text(ai_text):
    """
    Convert markdown-style formatting (##, **text**) into simple text with bold for subheadings.
    """
    cleaned_lines = []
    for line in ai_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading markdown headers like ## or #
        line = re.sub(r"^#+\s*", "", line)

        # Convert **bold** to <b>bold</b> for ReportLab
        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)

        cleaned_lines.append(line)
    return cleaned_lines

# ---------------------------
# AI Insights via Ollama
# ---------------------------
def generate_ai_description(data):
    prompt = f"""
Patient MRI Analysis Report
- Tumor Type: {data['tumor_type']}
- Model Confidence: {data['confidence']:.2f}%

Generate:
1) Doctor-friendly description.
2) Why the model predicts this type.
3) Suggested next steps.
4) Patient-friendly summary.
5) Common treatment pathways.
"""
    resp = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

# ---------------------------
# PDF Generator
# ---------------------------
def create_pdf(report_path, data, ai_text, images):
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("<b><font size=16 color=darkblue>🧠 MRI Brain Tumor Diagnostic Report</font></b>", styles["Title"]),
                Spacer(1,0.3*inch)]
    table = Table([
        ["Patient ID", data['patient_id']],
        ["Patient Name", data['patient_name']],
        ["Age", str(data['age'])],
        ["Scan Date", data['scan_date']],
        ["Tumor Type", data['tumor_type']],
        ["Model Confidence", f"{data['confidence']:.2f}%"]
    ], colWidths=[2.5*inch, 3.5*inch])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey)]))
    elements += [table, Spacer(1,0.3*inch),
                 Paragraph("<b>AI-Generated Medical Insights:</b>", styles["Heading2"]),
                 Spacer(1,0.1*inch)]
    # for line in ai_text.split("\n"): 
    #     if line.strip(): elements.append(Paragraph(line.strip(), styles["Normal"]))
    for line in clean_ai_text(ai_text):
        # If line looks like a subheading (ends with ":"), make it Heading3
        if line.endswith(":"):
            elements.append(Paragraph(f"<b>{line}</b>", styles["Heading3"]))
        else:
            elements.append(Paragraph(line, styles["Normal"]))

    elements.append(Spacer(1,0.3*inch))
    elements.append(Paragraph("<b>Imaging Results:</b>", styles["Heading2"]))
    for src, caption in images:
        buf = io.BytesIO()
        if isinstance(src, Image.Image): src.save(buf, format="PNG"); buf.seek(0); rl_img = RLImage(buf, width=5*inch, height=3.75*inch)
        else: rl_img = RLImage(src, width=5*inch, height=3.75*inch)
        elements += [rl_img, Paragraph(caption, styles["Italic"]), Spacer(1,0.2*inch)]
    elements.append(Paragraph("<b>Disclaimer:</b> AI-generated report, not a substitute for medical advice.Always consult a qualified healthcare provider for medical concerns.", styles["Normal"]))
    doc.build(elements)

# ---------------------------
# Flask Routes
# ---------------------------
model = load_ensemble_model()
transform = get_image_transform()

@app.route("/results/<filename>")
def results_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
