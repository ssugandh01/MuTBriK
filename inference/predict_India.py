"""
Nationwide brick kiln inference with a trained U-Net (EfficientNet-B7).

Runs sliding-window prediction (512 x 512 windows, 50% overlap by default) over
6-band Sentinel-2 GeoTIFF tiles (B2, B3, B4, B8, B11, B12), adds NDVI and NDBI,
averages the softmax outputs of overlapping windows, and writes one single-band
GeoTIFF of class IDs (0 = background, 1 = FCBTK, 2 = Zigzag, 3 = CBTK,
4 = RBTK, 5 = DDK) per input tile.

Example:
    python inference/predict_India.py \
        --input_dir  /path/to/sentinel2_tiles \
        --output_dir /path/to/predictions \
        --model_path /path/to/unet_b7_best_model.pth
"""
import os
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window
from segmentation_models_pytorch import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_image(image):
    """Per-channel min-max normalization, identical to the training pipeline."""
    image = np.nan_to_num(image, nan=0)
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    return ((image - min_val) / (max_val - min_val + 1e-6)).astype(np.float32)


def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)


def calculate_ndbi(swir1, nir):
    return (swir1 - nir) / (swir1 + nir + 1e-6)


def predict_tile(model, image_path, output_path, window_size=512, step=256, num_classes=6):
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

        prob_sum = np.zeros((num_classes, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, step):
            for x in range(0, width, step):
                window = Window(x, y, min(window_size, width - x), min(window_size, height - y))
                image = src.read(window=window).astype(np.float32)  # (6, h, w)

                red, nir, swir1 = image[2], image[3], image[4]
                image = np.vstack((image,
                                   calculate_ndvi(red, nir)[np.newaxis],
                                   calculate_ndbi(swir1, nir)[np.newaxis]))
                image = normalize_image(image)

                padded = np.zeros((image.shape[0], window_size, window_size), dtype=np.float32)
                padded[:, :image.shape[1], :image.shape[2]] = image
                tensor = torch.from_numpy(padded).unsqueeze(0).to(device)

                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    probs = F.softmax(model(tensor), dim=1).squeeze(0).float().cpu().numpy()

                h, w = window.height, window.width
                prob_sum[:, y:y + h, x:x + w] += probs[:, :h, :w]
                counts[y:y + h, x:x + w] += 1

    counts[counts == 0] = 1
    prediction = np.argmax(prob_sum / counts, axis=0).astype(np.uint8)

    meta.update({"dtype": "uint8", "count": 1})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(prediction, 1)


def load_model(model_path, num_classes):
    model = Unet("efficientnet-b7", encoder_weights=None, in_channels=8, classes=num_classes, activation=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description="Nationwide brick kiln inference on Sentinel-2 tiles")
    p.add_argument("--input_dir", required=True,
                   help="Folder with 6-band Sentinel-2 GeoTIFFs, directly or in subfolders")
    p.add_argument("--output_dir", required=True, help="Where predicted GeoTIFFs are written")
    p.add_argument("--model_path", required=True, help="Trained model checkpoint (.pth)")
    p.add_argument("--subfolder", default=None, help="Only process this subfolder of input_dir")
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--window_size", type=int, default=512)
    p.add_argument("--step", type=int, default=256, help="256 = 50%% overlap for 512 windows")
    args = p.parse_args()

    print(f"Using device: {device}")
    model = load_model(args.model_path, args.num_classes)

    search_root = os.path.join(args.input_dir, args.subfolder) if args.subfolder else args.input_dir
    image_paths = sorted(glob(os.path.join(search_root, "*.tif")) +
                         glob(os.path.join(search_root, "*", "*.tif")))
    if not image_paths:
        raise SystemExit(f"No .tif files found in {search_root}")

    for i, image_path in enumerate(image_paths, start=1):
        relative_dir = os.path.relpath(os.path.dirname(image_path), args.input_dir)
        out_dir = os.path.join(args.output_dir, relative_dir)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_predicted.tif")

        print(f"[{i}/{len(image_paths)}] {image_path}")
        predict_tile(model, image_path, out_path, args.window_size, args.step, args.num_classes)


if __name__ == "__main__":
    main()
