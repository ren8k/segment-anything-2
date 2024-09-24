import json
import math

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw


# use from https://github.com/wkentaro/labelme/blob/main/labelme/utils/shape.py#L21
def shape_to_mask(
    img_shape: tuple,
    points: list,
    shape_type: str = None,
    line_width: int = 10,
    point_size: int = 5,
) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def main() -> None:
    mask_json_path = "./dogcat_v2.json"

    with open(mask_json_path, "r", encoding="utf-8") as f:
        df = json.load(f)
    # dj['shapes'][0]は今回一つのラベルのため。
    mask = shape_to_mask(
        (df["imageHeight"], df["imageWidth"]),
        df["shapes"][0]["points"],
        shape_type=df["shapes"][0]["shape_type"],
        line_width=1,  # dummy
        point_size=1,  # dummy
    )

    # マスクを画像として保存
    mask = ~mask
    mask_image = PIL.Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save("mask_image.png")

    # マスクを直接表示
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title(f"Mask Image (Size: {mask.shape[1]}x{mask.shape[0]})")
    plt.show()

    print("Mask image has been saved as 'mask_image.png' and displayed.")


if __name__ == "__main__":
    main()
