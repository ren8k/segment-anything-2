import utils
from image_generator import ImageGenerator


def main() -> None:
    region = "us-west-2"
    model_id = "amazon.titan-image-generator-v2:0"
    input_image_path = "../images/dogcat.png"
    save_masks_dir = "../masks/dogcat"
    mask_image_path = f"{save_masks_dir}/mask_1.png"
    prompt = "A black cat"
    negative_prompt = "bad quality, low res, noise"
    mask_prompt = "A white cat"

    img_generator = ImageGenerator(region=region)
    input_image = utils.read_image_as_base64(input_image_path)
    mask_image = utils.read_image_as_base64(mask_image_path)

    img_generator.make_inpaint_payload(
        prompt, negative_prompt, input_image, mask_image=mask_image
    )
    # img_generator.make_payload(
    #     prompt, negative_prompt, input_image, mask_prompt=mask_prompt
    # )
    images = img_generator.generate_image(model_id)
    for img in images:
        img.show()
    # utils.show_image(image_bytes)


if __name__ == "__main__":
    main()
