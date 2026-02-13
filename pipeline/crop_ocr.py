"""
Crop images using a blueprint JSON and perform OCR on each crop.
Saves cropped images and outputs structured OCR results to JSON.
"""

import os
import json
from pathlib import Path
import cv2
import argparse

from ocr_dbg.ocr_engine import recognize_text_from_crop  # your existing OCR function


# ------------------ Utility ------------------

def load_blueprint(json_path):
    """
    Load blueprint JSON containing bounding boxes.
    Assumes coordinates are percentages (0–100).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Adjust this if your JSON structure differs
    return data[0]["bbox"]


def percent_to_pixels(box, img_w, img_h):
    x = int((box["x"] / 100) * img_w)
    y = int((box["y"] / 100) * img_h)
    w = int((box["width"] / 100) * img_w)
    h = int((box["height"] / 100) * img_h)
    return x, y, w, h


# ------------------ Crop Engine ------------------

def crop_regions_from_template(image_np, bbox_template):
    img_h, img_w = image_np.shape[:2]
    crops = {}

    for entry in bbox_template:
        name = entry.get("name") or entry.get("field") or f"field_{len(crops)}"

        x, y, w, h = percent_to_pixels(entry, img_w, img_h)

        # Clamp inside image
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            continue

        crop = image_np[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        crops[name] = crop

    return crops


# ------------------ OCR Stage ------------------

def ocr_crops(crop_dict, image_name, save_root="saved_crops"):
    """
    Saves crops to disk and performs OCR.
    Returns structured dictionary.
    """
    image_stem = Path(image_name).stem
    crop_dir = Path(save_root) / image_stem
    crop_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for field_name, crop_img in crop_dict.items():
        try:
            # Save crop
            crop_path = crop_dir / f"{field_name}.png"
            cv2.imwrite(str(crop_path), crop_img)

            # OCR
            text = recognize_text_from_crop(crop_img)
            clean_text = text.replace("\n", " ").strip()

            results[field_name] = {
                "ocr_text": clean_text,
                "crop_path": str(crop_path)
            }

        except Exception as e:
            print(f"OCR failed for {field_name}: {e}")
            results[field_name] = {
                "ocr_text": "",
                "crop_path": None
            }

    return results


# ------------------ Main Processing ------------------

def process_images(input_paths, blueprint_path, output_json="ocr_results.json"):
    bbox_template = load_blueprint(blueprint_path)

    all_results = []

    for input_path in input_paths:
        p = Path(input_path)

        if p.is_dir():
            files = sorted([
                str(x) for x in p.glob("*")
                if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
            ])
        else:
            files = [str(p)]

        for file_path in files:
            print(f"\nProcessing: {file_path}")

            img = cv2.imread(file_path)
            if img is None:
                print(f"Failed to read {file_path}")
                continue

            crops = crop_regions_from_template(img, bbox_template)
            ocr_results = ocr_crops(crops, file_path)

            all_results.append({
                "_source_file": file_path,
                "fields": ocr_results
            })

    # Save structured JSON
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, indent=2)

    print(f"\n OCR results saved to {output_json}")


# ------------------ CLI ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="Input image directory")
    parser.add_argument("--blueprint", required=True,
                        help="Blueprint JSON file")
    parser.add_argument("--output", default="ocr_results.json",
                        help="Output JSON file")

    args = parser.parse_args()

    process_images(args.input, args.blueprint, args.output)


