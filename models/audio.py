import torch
from transformers import AutoProcessor, AutoModel, BarkProcessor, BarkModel
import numpy as np
from schemas import VoicePresets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_mode() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained(
        "suno/bark-small",
        device=device
    )
    model = AutoModel.from_pretrained(
        "suno/bark-small",
        device=device
    )
    return processor, model

def generate_audio(
        processor: BarkProcessor,
        model: BarkModel,
        prompt: str,
        preset: VoicePresets
) -> tuple[np.array, int]:
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        voice_preset=preset
    )

    output = model.generate(
        **inputs,
        do_sample=True,
    ).cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    return output, sample_rate