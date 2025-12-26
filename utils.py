import io
from PIL import Image

def pil_from_bytes(b: bytes):
    return Image.open(io.BytesIO(b)).convert('RGB')
