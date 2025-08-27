import torch

ckpt = torch.load("hybrid_cnn_vit_best.pt", map_location="cpu")

# If it's a dict, print top-level keys
if isinstance(ckpt, dict):
    print("Top-level keys:", ckpt.keys())

    # If it has 'state_dict', check what keys are inside
    if "state_dict" in ckpt:
        print("State dict sample keys:", list(ckpt["state_dict"].keys())[:20])
    else:
        print("Sample keys:", list(ckpt.keys())[:20])
else:
    print("Checkpoint type:", type(ckpt))
