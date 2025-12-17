import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from numpy.typing import NDArray
import skimage # type: ignore
import inspect
from skimage.filters import gaussian, laplace # type: ignore

def _estimate_noise_sigma(image: NDArray[np.float32]) -> float:
    """
    Estimate the noise in an image using the sigma method
    Args:
        image: [H, W, C]

    Returns:
        The estimated noise sigma
    """
    sigma = skimage.restoration.estimate_sigma(image, average_sigmas=True, channel_axis=-1)

    return float(sigma)


def _estimate_noise_laplacian(image: NDArray[np.float32]) -> float:
    """
    Estimate the noise in an image using the Laplacian method.
    Args:
        image: [H,W,C]
    Returns:
        float: Estimated noise level in the input image.
    """
    filtered_image = laplace(image, ksize=3)
    sigma = np.mean(np.abs(filtered_image))
    return float(sigma)


def estimate_noise(image: NDArray[np.float32], method: str = 'sigma') -> float:
    """
    Estimate the noise in an image using a specified method.
    Args:
        image: Input image as a 2D or 3D numpy array.
        method: Method to use for noise estimation. Supported methods are 'sigma' and 'laplacian'.
    Returns:
        float: Estimated noise level in the input image.
    """
    validate_image(image)
    if method == 'sigma':
        return _estimate_noise_sigma(image)
    elif method == 'laplacian':
        return _estimate_noise_laplacian(image)
    else:
        raise ValueError(f"Unsupported noise estimation method: {method}")


def validate_image(image: NDArray[np.float32], expected_channels: Optional[int] = None) -> None:
    """
        Validate the input image by checking its type, dimensions, and data type.

        Args:
            image (NDArray): The image to validate.
            expected_channels (int): The expected number of color channels (e.g., 1 for grayscale, 3 for RGB).

        Raises:
            NotImplementedError: If the input is not a NumPy array.
            ValueError: If the image does not have the expected number of dimensions or channels.
            Exception: If the image data type is not one of the allowed types.
        """

    caller_frame = inspect.stack()[1]
    caller_name = caller_frame.function

    if not isinstance(image, np.ndarray):
        raise NotImplementedError(
            f"Wrong input type sent to metadata {caller_name}: Expected numpy array Got {type(image)}.")

    if image.dtype.name != 'float32':
        raise Exception(
            f"Wrong input type sent to metadata {caller_name}: Expected dtype float32 Got {image.dtype.name}.")

    if image.ndim != 3:
        raise ValueError(f"Wrong input dimension sent to metadata {caller_name}: Expected 3D but Got {image.ndim}D.")

    if expected_channels and expected_channels != image.shape[-1]:
        raise ValueError(
            f"Wrong input dimension sent to metadata {caller_name}: Expected {expected_channels} channels, "
            f"but Got {image.shape[-1]} channels.")


def detect_sharpness(image: NDArray[np.float32]) -> float:
    """
    Get an image in shape (H,W,C) and return a sharpness metric based on the gradient magnitude.

    Args: image (NDArray[np.float32]): A gray scale image represented as a NumPy array.

    Returns:
        Dict[str, np.float32]: A dictionary containing:
            - 'sharpness': The average gradient magnitude, representing the sharpness of the image.

    Description:
        This function computes the gradient magnitude using the Sobel operator in both the x and y directions.
        The sharpness metric is determined by calculating the mean of the gradient magnitude, which quantifies the
        overall sharpness of the image.
        The sharpness value is rounded to two decimal places before being returned.
    """
    validate_image(image)

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.round(np.mean(gradient_magnitude), 2))
