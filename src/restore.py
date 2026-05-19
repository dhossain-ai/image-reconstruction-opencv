import cv2
import os


INPUT_IMAGE_PATH = "images/input/damaged_photo.jpg"
OUTPUT_DIR = "images/output"
GRAYSCALE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "grayscale_photo.jpg")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = cv2.imread(INPUT_IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image from {INPUT_IMAGE_PATH}")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(GRAYSCALE_OUTPUT_PATH, grayscale)

    print(f"Grayscale image saved to: {GRAYSCALE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()