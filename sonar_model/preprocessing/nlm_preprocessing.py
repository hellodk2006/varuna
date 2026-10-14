import cv2
import numpy as np


def nlm_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means denoising to a sonar image.

    Geometry is preserved:
    - No resizing
    - No cropping
    - No rotation

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    np.ndarray
        NLM-denoised BGR image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )


def preprocess_image(input_path: str, output_path: str) -> None:
    """
    Read an image, apply NLM preprocessing, and save the result.
    """

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    processed = nlm_preprocess(image)

    success = cv2.imwrite(output_path, processed)

    if not success:
        raise IOError(f"Could not write image: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply NLM preprocessing to a sonar image."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input sonar image."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for preprocessed image."
    )

    args = parser.parse_args()

    preprocess_image(args.input, args.output)

    print(f"Preprocessed image saved to: {args.output}")