import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_IMAGE_PATH = BASE_DIR / "images" / "input" / "damaged_photo.jpg"
OUTPUT_DIR = BASE_DIR / "images" / "output"
MASK_DIR = BASE_DIR / "masks"

GRAYSCALE_OUTPUT_PATH = OUTPUT_DIR / "grayscale_photo.jpg"
CONTRAST_OUTPUT_PATH = OUTPUT_DIR / "contrast_enhanced_photo.jpg"
DENOISED_OUTPUT_PATH = OUTPUT_DIR / "denoised_photo.jpg"
INPAINTED_OUTPUT_PATH = OUTPUT_DIR / "inpainted_photo.jpg"

MASK_OUTPUT_PATH = MASK_DIR / "scratch_crack_mask.jpg"


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    return image


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(grayscale_image):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(grayscale_image)


def reduce_noise(image):
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )


def generate_damage_mask(image):
    # Detect very bright damaged areas such as scratches and torn photo marks
    _, bright_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Remove small noise from the mask
    kernel_small = np.ones((2, 2), np.uint8)
    cleaned_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # Connect broken scratch/crack regions
    kernel_line = np.ones((3, 3), np.uint8)
    refined_mask = cv2.dilate(
        cleaned_mask,
        kernel_line,
        iterations=1
    )

    return refined_mask


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    image = load_image(INPUT_IMAGE_PATH)

    grayscale = convert_to_grayscale(image)
    contrast_enhanced = enhance_contrast(grayscale)
    denoised = reduce_noise(contrast_enhanced)
    damage_mask = generate_damage_mask(denoised)
    inpainted = inpaint_damage(denoised, damage_mask)

    cv2.imwrite(str(GRAYSCALE_OUTPUT_PATH), grayscale)
    cv2.imwrite(str(CONTRAST_OUTPUT_PATH), contrast_enhanced)
    cv2.imwrite(str(DENOISED_OUTPUT_PATH), denoised)
    cv2.imwrite(str(MASK_OUTPUT_PATH), damage_mask)
    cv2.imwrite(str(INPAINTED_OUTPUT_PATH), inpainted)

    print(f"Grayscale image saved to: {GRAYSCALE_OUTPUT_PATH}")
    print(f"Contrast enhanced image saved to: {CONTRAST_OUTPUT_PATH}")
    print(f"Denoised image saved to: {DENOISED_OUTPUT_PATH}")
    print(f"Damage mask saved to: {MASK_OUTPUT_PATH}")
    print(f"Inpainted image saved to: {INPAINTED_OUTPUT_PATH}")

def inpaint_damage(image, mask):
    return cv2.inpaint(
        image,
        mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )    


if __name__ == "__main__":
    main()