import utils
from image_generator import ImageGenerator


def main() -> None:
    region = "us-west-2"
    model_id = "amazon.titan-image-generator-v2:0"
    input_image_path = "../images/dogcat.png"
    save_masks_dir = "../masks/dogcat"
    # mask_image_path = f"{save_masks_dir}/mask_1.png"
    mask_image_path = f"{save_masks_dir}/mask_dog_rectangle.png"
    # prompt = "A black cat inside a red bucket, background is dim green nature"
    # prompt = "A dog riding in a small boat."
    prompt = "A cute brown puppy and a white cat inside a blue bucket, an orange sunset in the evening"
    # prompt = "A smiling dog and cat inside a red bucket"
    negative_prompt = "bad quality, low res, noise"
    mask_prompt = "A cute brown puppy"
    seed = 17  # removeは7が良い．
    num_image = 5

    img_generator = ImageGenerator(region=region)
    input_image = utils.read_image_as_base64(input_image_path)
    input_images = [input_image]
    mask_image = utils.read_image_as_base64(mask_image_path)

    # payload = img_generator.make_outpaint_payload(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     input_image=input_image,
    #     # mask_image=mask_image,
    #     mask_prompt=mask_prompt,
    #     outpainting_mode="PRECISE",
    # )
    # payload = img_generator.make_variation_payload(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     input_images=input_images,
    #     similarity_strength=0.7,
    # )
    payload = img_generator.make_color_guide_payload(
        prompt=prompt,
        negative_prompt=negative_prompt,
        colors=["#FFA500", "#87CEEB"],
        reference_image=input_image,
    )
    # payload = img_generator.make_background_removal_payload(input_image=input_image)
    response = img_generator.generate_image(
        payload=payload, model_id=model_id, seed=seed, num_image=num_image
    )
    images = img_generator.extract_images_from(response)
    utils.save_images(images, seed=seed)
    utils.show_images(images)


if __name__ == "__main__":
    main()
