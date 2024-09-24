import base64
import io
import json
from typing import Any, Optional

import boto3
from botocore.client import BaseClient
from PIL import Image


class ImageGenerator:
    def __init__(self, region: str) -> None:
        self.client = self._init_bedrock_client(region)

    def _init_bedrock_client(self, region: str) -> BaseClient:
        return boto3.client(service_name="bedrock-runtime", region_name=region)

    def make_payload(self, mode: str, **kwargs: Any) -> dict:
        if mode == "TEXT_IMAGE":
            return self.make_text_to_image_payload(**kwargs)
        else:
            raise ValueError("mode is not supported")

    def make_text_to_image_payload(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> dict:
        return {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt,
            },
        }

    def make_image_conditioning_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_image: str,
        control_mode: str,
        control_strength: float = 0.7,
    ) -> dict:
        if not (control_mode == "CANNY_EDGE" or control_mode == "SEGMENTATION"):
            raise ValueError("control_mode must be either CANNY_EDGE or SEGMENTATION")
        return {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": negative_prompt,
                "conditionImage": input_image,
                "controlMode": control_mode,
                "controlStrength": control_strength,
            },
        }

    def make_inpaint_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_image: str,
        mask_image: Optional[str] = None,
        mask_prompt: Optional[str] = None,
    ) -> dict:
        if mask_image:
            mask_data = {"maskImage": mask_image}
        elif mask_prompt:
            mask_data = {"maskPrompt": mask_prompt}
        else:
            raise ValueError("Either mask_image or mask_prompt must be provided")
        return {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "text": prompt,
                "negativeText": negative_prompt,
                "image": input_image,
                **mask_data,
            },
        }

    def make_object_removal_payload(
        self,
        negative_prompt: str,
        input_image: str,
        mask_image: Optional[str] = None,
        mask_prompt: Optional[str] = None,
    ) -> dict:
        if mask_image:
            mask_data = {"maskImage": mask_image}
        elif mask_prompt:
            mask_data = {"maskPrompt": mask_prompt}
        else:
            raise ValueError("Either mask_image or mask_prompt must be provided")
        return {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "negativeText": negative_prompt,
                "image": input_image,
                **mask_data,
            },
        }

    def make_outpaint_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_image: str,
        mask_image: Optional[str] = None,
        mask_prompt: Optional[str] = None,
        outpainting_mode: str = "DEFAULT",
    ) -> dict:
        if mask_image:
            mask_data = {"maskImage": mask_image}
        elif mask_prompt:
            mask_data = {"maskPrompt": mask_prompt}
        else:
            raise ValueError("Either mask_image or mask_prompt must be provided")
        return {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "text": prompt,
                "negativeText": negative_prompt,
                "image": input_image,
                "outPaintingMode": outpainting_mode,  # One of "PRECISE" or "DEFAULT"
                **mask_data,
            },
        }

    def make_variation_payload(
        self,
        prompt: str,
        negative_prompt: str,
        input_images: list,  # ["base64-encoded string"]
        similarity_strength: float = 0.7,
    ) -> dict:
        return {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "text": prompt,
                "negativeText": negative_prompt,  # Optional
                "images": input_images,  # One image is required
                "similarityStrength": similarity_strength,  # Range: 0.2 to 1.0
            },
        }

    def make_color_guide_payload(
        self,
        prompt: str,
        negative_prompt: str,
        colors: list,  # ["#FFFFFF", "#000000"]
        reference_image: Optional[str] = None,
    ) -> dict:
        if reference_image:
            return {
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "text": prompt,
                    "negativeText": negative_prompt,
                    "referenceImage": reference_image,
                    "colors": colors,
                },
            }
        else:
            return {
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "text": prompt,
                    "negativeText": negative_prompt,
                    "colors": colors,
                },
            }

    def make_background_removal_payload(
        self,
        input_image: str,
    ) -> dict:
        return {
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {
                "image": input_image,
            },
        }

    def generate_image(
        self,
        payload: dict,
        model_id: str,
        num_image: int = 2,
        cfg_scale: float = 10.0,
        seed: int = 0,
    ) -> dict:
        body = json.dumps(
            {
                **payload,
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
