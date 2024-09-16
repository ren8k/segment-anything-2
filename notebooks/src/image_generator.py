import base64
import io
import json
from typing import Optional

import boto3
from botocore.client import BaseClient
from PIL import Image


class ImageGenerator:
    def __init__(self, region: str) -> None:
        self.client = self._init_bedrock_client(region)
        self.payload = {}

    def _init_bedrock_client(self, region: str) -> BaseClient:
        return boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

    def make_inpaint_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_image: str,
        mask_image: Optional[str] = None,
        mask_prompt: Optional[str] = None,
    ) -> None:
        if mask_image:
            mask_data = {"maskImage": mask_image}
        elif mask_prompt:
            mask_data = {"maskPrompt": mask_prompt}
        else:
            raise ValueError("Either mask_image or mask_prompt must be provided")
        self.payload = {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "text": prompt,
                "negativeText": negative_prompt,
                "image": input_image,
                **mask_data,
            },
        }

    def generate_image(
        self,
        model_id: str,
        num_image: int = 2,
        cfg_scale: float = 10.0,
        seed: int = 0,
    ) -> list:
        body = json.dumps(
            {
                **self.payload,
                "imageGenerationConfig": {
                    "numberOfImages": num_image,  # Range: 1 to 5
                    "quality": "premium",  # Options: standard/premium
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": cfg_scale,  # Range: 1.0 (exclusive) to 10.0
                    "seed": seed,  # Range: 0 to 214783647
                },
            }
        )
        accept = "application/json"
        content_type = "application/json"

        response = self.client.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )
        response_body = json.loads(response.get("body").read())
        images = [
            Image.open(io.BytesIO(base64.b64decode(base64_image)))
            for base64_image in response_body.get("images")
        ]

        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise f"Image generation error. Error is {finish_reason}"

        return images

        # base64_image = response_body.get("images")[0]
        # base64_bytes = base64_image.encode("utf-8")
        # image_bytes = base64.b64decode(base64_bytes)

        # return image_bytes
