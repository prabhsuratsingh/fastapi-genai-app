from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse

from models import load_text_model, load_audio_model, generate_text, generate_audio
from schemas import VoicePresets
from utils import audio_array_to_buffer

app = FastAPI()

@app.get("/generate/text")
def serve_lang_model_controller(prompt: str) -> str:
    pipe = load_text_model()
    output = generate_text(pipe, prompt)
    return output

@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1"
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(
        processor, 
        model, 
        prompt,
        preset
    )
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate),
        media_type="audio/wav"
    )





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)