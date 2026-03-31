import os
import json
import argparse
from torchvision.models import ResNet50_Weights

# Map class folder index (0-999) -> "a photo of a {label}"
LABELS = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def generate_image_meta(split_dir, output_path):
    """Generate {image_path, caption} metadata for token extraction input."""
    meta = []
    for entry in sorted(os.listdir(split_dir), key=lambda x: int(x) if x.isdigit() else -1):
        cls_path = os.path.join(split_dir, entry)
        if not os.path.isdir(cls_path) or not entry.isdigit():
            continue
        label = LABELS[int(entry)]
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                meta.append({
                    "image_path": os.path.join(cls_path, fname),
                    "caption": label,
                })
    with open(output_path, "w") as f:
        json.dump(meta, f)
    print(f"Saved {len(meta)} entries to {output_path}")


def generate_token_meta(tokens_dir, output_path, resolution):
    """Generate {code_path, label_path} metadata for training from extracted tokens."""
    code_dir = os.path.join(tokens_dir, f"{resolution}_codes")
    label_dir = os.path.join(tokens_dir, f"{resolution}_labels")
    meta = []
    for fname in os.listdir(code_dir):
        if not fname.endswith(".npy"):
            continue
        code_path = os.path.join(code_dir, fname)
        label_path = os.path.join(label_dir, fname)
        if os.path.exists(code_path) and os.path.exists(label_path):
            meta.append({"code_path": code_path, "label_path": label_path})
    with open(output_path, "w") as f:
        json.dump(meta, f)
    print(f"Saved {len(meta)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image_meta", "token_meta"], required=True,
                        help="image_meta: for token extraction input; token_meta: for training")
    parser.add_argument("--split_dir", help="Path to ImageNet split dir (image_meta mode)")
    parser.add_argument("--tokens_dir", help="Path to extracted tokens dir (token_meta mode)")
    parser.add_argument("--resolution", type=int, default=256, help="Token resolution to use (token_meta mode)")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    if args.mode == "image_meta":
        assert args.split_dir, "--split_dir required for image_meta mode"
        generate_image_meta(args.split_dir, args.output)
    else:
        assert args.tokens_dir, "--tokens_dir required for token_meta mode"
        generate_token_meta(args.tokens_dir, args.output, args.resolution)
