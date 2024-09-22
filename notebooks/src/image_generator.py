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
        self.payload: dict = {}

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

    def make_text_to_image_payload(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> None:
        self.payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt,
            },
        }

    def make_object_removal_payload(
        self,
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
                "negativeText": negative_prompt,
                "image": input_image,
                **mask_data,
            },
        }

    def make_image_conditioning_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_image: str,
        control_mode: str,
        control_strength: float = 0.7,
    ) -> None:
        if not (control_mode == "CANNY_EDGE" or control_mode == "SEGMENTATION"):
            raise ValueError("control_mode must be either CANNY_EDGE or SEGMENTATION")
        self.payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt,
                "conditionImage": input_image,
                "controlMode": control_mode,
                "controlStrength": control_strength,
            },
        }

    def generate_image(
        self,
        model_id: str,
        num_image: int = 2,
        cfg_scale: float = 10.0,
        seed: int = 0,
    ) -> dict:
        body = json.dumps(
            {
                **self.payload,
                "imageGenerationConfig": {
                    "numberOfImages": num_image,  # Range: 1 to 5
                    "quality": "premium",  # Options: standard/premium
                    "height": 1024,  # Supported height list in the docs
                    "width": 1024,  # Supported width list in the docs
                    "cfgScale": cfg_scale,  # Range: 1.0 (exclusive) to 10.0
                    "seed": seed,  # Range: 0 to 214783647
                },
            }
        )
        accept = "application/json"
        content_type = "application/json"

        return self.client.invoke_model(
            body=body, modelId=model_id, accept=accept, contentType=content_type
        )

    def extract_images_from(self, response: dict) -> list:
        response_body = json.loads(response.get("body").read())
        self._validate_response(response_body)
        images = [
            Image.open(io.BytesIO(base64.b64decode(base64_image)))
            for base64_image in response_body.get("images")
        ]
        return images

    def _validate_response(self, response_body: dict) -> None:
        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise Exception(f"Image generation error. Error is {finish_reason}")
