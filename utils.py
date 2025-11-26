from io import BytesIO
import soundfile
import numpy as np
from typing import Literal
from PIL import Image

def audio_array_to_buffer(
        audio_array: np.array,
        sample_rate: int
) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(
        buffer,
        audio_array,
        samplerate=sample_rate,
        format="wav"
    )
    buffer.seek(0)
    return buffer

def image_to_bytes(
        image: Image.Image,
        format: Literal["PNG", "JPEG"] = "PNG"
) -> BytesIO:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()