import base64
import io
from PIL import Image

import torchvision.transforms as transforms

from exceptions.PreProcessExceptions import Base64DecodeError, PreprocessError, UnsupportedInputTypeError

# ---- Image preprocessing ----


def preprocess(base64_str, metadata):

    # Define standard preprocessing steps
    resizeMatrix = tuple(metadata["input_size"])
    normalizeMean = metadata["mean"]
    normalizeStd = metadata["std"]

    transform = transforms.Compose([
        transforms.Resize(resizeMatrix[0]),
        transforms.ToTensor(),
        transforms.Normalize(normalizeMean, normalizeStd)
    ])

    # Decode base64 → bytes
    try:
        content_bytes = base64.b64decode(base64_str)
    except Exception as e:
        raise Base64DecodeError(e)
    
    if metadata["input_type"] == "image_2d":
        try:
            image = Image.open(io.BytesIO(content_bytes)).convert("RGB")
            return transform(image).unsqueeze(0)
        except Exception as e:
            raise PreprocessError(f"Failed to process image: {e}")
    else:
        raise UnsupportedInputTypeError(metadata["input_type"])

