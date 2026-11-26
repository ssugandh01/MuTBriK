import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from glob import glob
from segmentation_models_pytorch import Unet
import wandb  # minimal use

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --------- helpers (unchanged) ----------
def normalize_image(image):
    min_val = image.min(axis=(1, 2), keepdims=True)
    max_val = image.max(axis=(1, 2), keepdims=True)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-6)
    return normalized_image.astype(np.float32)

def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red + 1e-6)

def calculate_ndbi(swir1, nir):
    return (swir1 - nir) / (swir1 + 1e-6 + nir)

def windowed_inference_optimized(model, image_path, window_size=512, step=256, num_classes=6, output_meta=None, output_path=None):
    with rasterio.open(image_path) as src:
        meta = src.meta
        height, width = meta['height'], meta['width']

        full_mask = np.zeros((num_classes, height, width), dtype=np.float32)
        count_mask = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, step):
            for x in range(0, width, step):
                window = rasterio.windows.Window(x, y, min(window_size, width - x), min(window_size, height - y))
                image = src.read(window=window).astype(np.float32)  # (6, H, W)

                red, nir, swir1 = image[2], image[3], image[4]
                ndvi = calculate_ndvi(red, nir)
                ndbi = calculate_ndbi(swir1, nir)

                image = np.vstack((image, ndvi[np.newaxis, :, :], ndbi[np.newaxis, :, :]))
                image = normalize_image(image)

                padded = np.zeros((image.shape[0], window_size, window_size), dtype=np.float32)
                padded[:, :image.shape[1], :image.shape[2]] = image
                image_tensor = torch.from_numpy(padded).unsqueeze(0).to(device).float()

                with torch.no_grad(), torch.cuda.amp.autocast():
                    output = model(image_tensor)
                    output = F.softmax(output, dim=1)
                    pred = output.squeeze(0).cpu().numpy()

                pred_cropped = pred[:, :window.height, :window.width]
                full_mask[:, y:y+window.height, x:x+window.width] += pred_cropped
                count_mask[y:y+window.height, x:x+window.width] += 1

        count_mask[count_mask == 0] = 1
        final_prediction = full_mask / count_mask
        final_prediction_class = np.argmax(final_prediction, axis=0).astype(np.uint8)

        if output_path:
            output_meta.update({'dtype': 'uint8', 'count': 1, 'height': height, 'width': width})
            with rasterio.open(output_path, 'w', **output_meta) as dst:
                dst.write(final_prediction_class, 1)

        return final_prediction_class

def load_model(model_path, num_classes):
    model = Unet('efficientnet-b7', in_channels=8, classes=num_classes, activation=None).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model

# --------- minimal wandb + args ----------
def parse_and_init():
    p = argparse.ArgumentParser()
    p.add_argument('--test_data_dir', type=str, default="/cloud/wwu1/d_satdat/shared_satellite_data/processed_sentinel/india/gee_download/")
    p.add_argument('--output_dir', type=str, default="/scratch/tmp/ssugandh/version_dynamic/output_new_data/")
    p.add_argument('--model_path', type=str, default="/scratch/tmp/ssugandh/version_dynamic/New_data_model/unet_b7_best_model.pth")
    p.add_argument('--subfolder', type=str, default=None, help="relative to test_data_dir")
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--window_size', type=int, default=512)
    p.add_argument('--step', type=int, default=256)
    p.add_argument('--project', type=str, default="brick-type-seg")
    p.add_argument('--entity', type=str, default=None)
    p.add_argument('--run_name', type=str, default=None)
    args = p.parse_args()

    # init wandb (no logs, no manual summary)
    run_name = args.run_name or args.subfolder or os.path.basename(args.test_data_dir.rstrip('/'))
    wandb.init(project=args.project, entity=args.entity, name=run_name, config=vars(args))
    return wandb.config

def main():
    cfg = parse_and_init()

    model = load_model(cfg.model_path, cfg.num_classes)
    model.eval()

    test_data_dir = cfg.test_data_dir
    if cfg.subfolder is not None:
        folders = [cfg.subfolder]
    else:
        folders = [f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))]

    test_image_paths = []
    for folder in folders:
        folder_path = os.path.join(test_data_dir, folder)
        test_image_paths.extend(glob(os.path.join(folder_path, "*.tif")))

    for i, image_path in enumerate(test_image_paths, start=1):
        filename = os.path.basename(image_path)
        relative_dir = os.path.relpath(os.path.dirname(image_path), test_data_dir)

        out_dir = os.path.join(cfg.output_dir, relative_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{filename}_predicted.tif')

        print(f"[{i}/{len(test_image_paths)}] {image_path}")
        with rasterio.open(image_path) as src:
            output_meta = src.meta

        _ = windowed_inference_optimized(
            model=model,
            image_path=image_path,
            window_size=cfg.window_size,
            step=cfg.step,
            num_classes=cfg.num_classes,
            output_meta=output_meta,
            output_path=out_path
        )

    # optional: wandb.finish()  # not required; included here only for cleanliness

if __name__ == "__main__":
    main()
