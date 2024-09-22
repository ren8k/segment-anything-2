import base64
import datetime

import matplotlib.pyplot as plt
import pytz
from PIL import Image


def read_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def get_datetime() -> str:
    tokyo_tz = pytz.timezone("Asia/Tokyo")
    return datetime.datetime.now(tokyo_tz).strftime("%Y%m%d_%H%M%S")


def save_images(images: list[Image.Image], seed: int) -> None:
    datetime = get_datetime()
    for i, image in enumerate(images):
        image.save(f"../results/output_{datetime}_{seed=}_{i}.png")


def show_images(images: list[Image.Image]) -> None:
    if len(images) == 1:
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0])
        plt.axis("off")
    else:
        # 一覧にして表示
        fig, axes = plt.subplots(1, len(images), figsize=(20, 10))
        for ax, image in zip(axes, images):
            ax.imshow(image)
            ax.axis("off")
        # plt.tight_layout()
    plt.show()
