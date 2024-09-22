import utils
from image_generator import ImageGenerator


def main() -> None:
    region = "us-west-2"
    model_id = "amazon.titan-image-generator-v2:0"
    input_image_path = "../images/dogcat.png"
    save_masks_dir = "../masks/dogcat"
    mask_image_path = f"{save_masks_dir}/mask_1.png"
    prompt = "A black cat inside a red bucket, background is dim green nature"
    # prompt = "two dogs walking down an urban street, facing the camera"
    negative_prompt = "deformed ears, deformed eyes, bad quality, low res, noise"
    mask_prompt = "A brown puppy"
    seed = 7  # removeは7が良い．
    num_image = 5

    img_generator = ImageGenerator(region=region)
    input_image = utils.read_image_as_base64(input_image_path)
    mask_image = utils.read_image_as_base64(mask_image_path)

    # img_generator.make_text_to_image_payload(prompt, negative_prompt)

    # img_generator.make_object_removal_payload(
    #     negative_prompt, input_image, mask_image=mask_image
    # )
    # img_generator.make_image_conditioning_payload(
    #     prompt, negative_prompt, input_image, control_mode="SEGMENTATION"
    # )
    # img_generator.make_object_removal_payload(
    #     negative_prompt, input_image, mask_prompt=mask_prompt
    # )
    img_generator.make_inpaint_payload(
        prompt, negative_prompt, input_image, mask_image=mask_image
    )
    # img_generator.make_payload(
    #     prompt, negative_prompt, input_image, mask_prompt=mask_prompt
    # )
    response = img_generator.generate_image(
        model_id=model_id, seed=seed, num_image=num_image
    )
    images = img_generator.extract_images_from(response)
    utils.save_images(images, seed=seed)
    utils.show_images(images)


if __name__ == "__main__":
    main()
