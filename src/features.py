"""Frozen-backbone feature extraction and CLIP zero-shot classification."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# name -> (library, model id). DINOv2 runs at 224px so every backbone sees the same resolution.
BACKBONES = {
    "resnet50":   ("timm", "resnet50.a1_in1k"),
    "vit_b16":    ("timm", "vit_base_patch16_224.augreg2_in21k_ft_in1k"),
    "dinov2_s14": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2_b14": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "clip_b16":   ("open_clip", "ViT-B-16/openai"),
}

CROP_NAMES = {"bellpepper": "bell pepper"}


class ImageList(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.transform(Image.open(self.samples[i].path).convert("RGB"))


def load_backbone(name, device):
    """Returns (model, preprocess, encode_fn)."""
    lib, model_id = BACKBONES[name]
    if lib == "timm":
        import timm

        kwargs = {"img_size": 224} if "dinov2" in model_id else {}
        model = timm.create_model(model_id, pretrained=True, num_classes=0, **kwargs)
        cfg = timm.data.resolve_data_config({}, model=model)
        preprocess = timm.data.create_transform(**cfg)
        encode = model
    else:
        import open_clip

        arch, tag = model_id.split("/")
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
        encode = model.encode_image
    return model.eval().to(device), preprocess, encode


@torch.no_grad()
def extract(name, samples, device="cuda", batch_size=128, num_workers=2, cache=None):
    """L2-normalised features, shape (len(samples), dim). Cached as .npy when `cache` is given."""
    if cache and Path(cache).exists():
        return np.load(cache)
    model, preprocess, encode = load_backbone(name, device)
    loader = DataLoader(ImageList(samples, preprocess), batch_size=batch_size, num_workers=num_workers)
    feats = []
    for x in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = encode(x.to(device))
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    feats = torch.cat(feats).numpy()
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, feats)
    return feats


def class_prompt(key):
    """'tomato_early_blight' -> 'a photo of a tomato leaf with early blight'."""
    crop, _, disease = key.partition("_")
    crop = CROP_NAMES.get(crop, crop)
    if disease == "healthy":
        return f"a photo of a healthy {crop} leaf"
    return f"a photo of a {crop} leaf with {disease.replace('_', ' ')}"


@torch.no_grad()
def clip_zero_shot(image_feats, class_keys, device="cuda"):
    """Predicted labels from CLIP text prompts; `image_feats` must come from clip_b16."""
    import open_clip

    arch, tag = BACKBONES["clip_b16"][1].split("/")
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.eval().to(device)
    text = model.encode_text(tokenizer([class_prompt(k) for k in class_keys]).to(device))
    text = torch.nn.functional.normalize(text.float(), dim=-1).cpu().numpy()
    return (image_feats @ text.T).argmax(axis=1)
