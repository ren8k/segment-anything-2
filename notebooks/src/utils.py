import base64
import io

import matplotlib.pyplot as plt
from PIL import Image


def read_image_as_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def show_image(image_bytes: bytes) -> None:
    image = Image.open(io.BytesIO(image_bytes))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
