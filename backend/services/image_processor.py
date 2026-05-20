"""
Image preprocessing service for Smart Lab Chair Monitoring System.
Handles image resizing, noise reduction, brightness normalization, and edge enhancement.
"""

import cv2
import numpy as np
from PIL import Image
import io
import os


class ImageProcessor:
    """Preprocesses uploaded images for optimal AI detection."""

    MAX_DIMENSION = 1280  # Max width or height in pixels
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}

    @staticmethod
    def validate_format(filename: str) -> bool:
        """Check if the file format is supported."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in ImageProcessor.SUPPORTED_FORMATS

    @staticmethod
    def read_image(image_bytes: bytes) -> np.ndarray:
        """Read image from bytes into OpenCV format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image. File may be corrupted.")
        return image

    @staticmethod
    def resize_image(image: np.ndarray) -> np.ndarray:
        """Resize image to max dimension while preserving aspect ratio."""
        h, w = image.shape[:2]
        max_dim = ImageProcessor.MAX_DIMENSION

        if max(h, w) <= max_dim:
            return image

        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def reduce_noise(image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter (preserves edges better than Gaussian)."""
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised

    @staticmethod
    def normalize_brightness(image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Applied to the L channel in LAB color space to preserve color.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    @staticmethod
    def enhance_edges(image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.
        This helps the ML model detect chair/desk boundaries more clearly.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    @staticmethod
    def preprocess(image_bytes: bytes) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Decode image
        2. Resize to max dimension
        3. Reduce noise
        4. Normalize brightness
        5. Enhance edges
        """
        image = ImageProcessor.read_image(image_bytes)
        image = ImageProcessor.resize_image(image)
        image = ImageProcessor.reduce_noise(image)
        image = ImageProcessor.normalize_brightness(image)
        image = ImageProcessor.enhance_edges(image)
        return image

    @staticmethod
    def save_image(image: np.ndarray, path: str, quality: int = 95) -> str:
        """Save image to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return path

    @staticmethod
    def encode_image(image: np.ndarray, fmt: str = ".jpg") -> bytes:
        """Encode image to bytes."""
        if fmt == ".png":
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
