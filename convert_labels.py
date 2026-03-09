import os
import cv2

CLASS_MAP = {
    "cavity": 0,
    "normal": 1
}

def convert_one(txt_path, img_path, save_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    yolo_lines = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            coords = list(map(float, parts[:8]))
            cls_name = parts[8]

            xs = coords[0::2]
            ys = coords[1::2]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_c = (x_min + x_max) / 2 / w
            y_c = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            cls_id = CLASS_MAP[cls_name]

            yolo_lines.append(
                f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
            )

    with open(save_path, "w") as f:
        f.write("\n".join(yolo_lines))


def process_split(split):
    img_dir = f"{split}/images"
    lbl_dir = f"{split}/labelTxt"
    out_dir = f"{split}/labels"

    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue

        txt_path = os.path.join(lbl_dir, file)
        img_name = file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        save_path = os.path.join(out_dir, file)
        convert_one(txt_path, img_path, save_path)


for s in ["train", "valid", "test"]:
    process_split(s)

print("✅ Label conversion finished.")