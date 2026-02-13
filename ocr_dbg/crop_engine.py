import cv2
import os

def percent_to_pixels(box, img_w, img_h):
    x = int((box["x"] / 100) * img_w)
    y = int((box["y"] / 100) * img_h)
    w = int((box["width"] / 100) * img_w)
    h = int((box["height"] / 100) * img_h)
    return x, y, w, h


def crop_fields(image_path, bboxes, save_dir="debug_crops"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_h, img_w = img.shape[:2]
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for idx, box in enumerate(bboxes):
        x, y, w, h = percent_to_pixels(box, img_w, img_h)

        # keep crop inside image
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            continue

        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        field = box["labels"][0]
        cv2.imwrite(f"{save_dir}/{idx}_{field}.png", crop)
        results.append((field, crop))

    print(f"Cropped {len(results)} fields")
    return results
