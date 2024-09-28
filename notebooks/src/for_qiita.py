import base64
import io
import json

import boto3
from PIL import Image


def generate_image(
    payload: dict,
    num_image: int = 1,
    cfg_scale: float = 10.0,
    seed: int = 42,
    model_id: str = "amazon.titan-image-generator-v2:0",
) -> None:

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    body = json.dumps(
        {
            **payload,
            "imageGenerationConfig": {
                "numberOfImages": num_image,  # Range: 1 to 5
                "quality": "premium",  # Options: standard/premium
                "height": 1024,  # Supported height list above
                "width": 1024,  # Supported width list above
                "cfgScale": cfg_scale,  # Range: 1.0 (exclusive) to 10.0
                "seed": seed,  # Range: 0 to 214783647
            },
        }
    )

    response = client.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    base64_image = response_body.get("images")[0]
    base64_bytes = base64_image.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    image = Image.open(io.BytesIO(image_bytes))
    image.show()


generate_image(
    {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": "A cute brown puppy and a white cat inside a red bucket",  # Required
            "negativeText": "bad quality, low res, noise",  # Optional
        },
    }
)
