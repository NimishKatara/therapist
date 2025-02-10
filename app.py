from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import whisper
from gtts import gTTS
from flask import Flask
from pydub import AudioSegment
from google.cloud import texttospeech
import requests
import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GEMINI_API_KEY or not SUPABASE_KEY or not SUPABASE_URL:
    raise ValueError("Missing API Keys. Check your .env file.")

# FastAPI App
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper Model
model = whisper.load_model("tiny")
print("✅ Whisper model loaded successfully!")

# Google Gemini API Config
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# Supabase Setup
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Therapist is running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)

@app.get("/")
async def root():
    # return {"message": "API is running with Gemini AI!"}
    return FileResponse("intro.mp3", media_type="audio/mpeg")

@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    """Handles audio input, transcribes it, gets AI response, and converts it back to speech."""
    try:
        audio_path = "input.wav"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        result = model.transcribe(audio_path)
        user_input = result["text"].strip()

        if not user_input:
            return JSONResponse(content={"error": "No speech detected"}, status_code=400)

        ai_response = get_gemini_response(user_input)

        if not ai_response:
            return JSONResponse(content={"error": "Failed to get AI response"}, status_code=500)

        audio_output_path = generate_speech(ai_response)

        return {
            "message": user_input,
            "response": ai_response,
            "audio_url": f"http://127.0.0.1:8000/audio/{audio_output_path}"
        }
    except Exception as e:
        print(f"🔥 Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat/text")
async def chat_text(input_text: str = Form(...)):
    """Handles text input and gets AI response with an optional audio reply."""
    try:
        input_text = input_text.strip()
        if not input_text:
            return JSONResponse(content={"error": "Empty message"}, status_code=400)

        ai_response = get_gemini_response(input_text)

        if not ai_response:
            return JSONResponse(content={"error": "Failed to get AI response"}, status_code=500)

        audio_output_path = generate_speech(ai_response)

        return {
            "message": input_text,
            "response": ai_response,
            "audio_url": f"http://127.0.0.1:8000/audio/{audio_output_path}"
        }
    except Exception as e:
        print(f"🔥 Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serves the generated audio file."""
    file_path = os.path.abspath(filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    return FileResponse(file_path, media_type="audio/mpeg")

def get_gemini_response(user_input: str) -> str:
    """Calls Gemini API and returns response text."""
    try:
        payload = {
            "contents": [{"parts": [{"text": f"Hey, be a chill Gen Z therapist and keep it short and your name is Suhana : {user_input}"}]}],
            "generationConfig": {"maxOutputTokens": 50}  # Specify token limit
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            return "Error: Gemini API is not responding."

        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
    
    except Exception as e:
        print(f"🔥 Gemini API Error: {e}")
        return "Oops, something went wrong."


def generate_speech(text: str) -> str:
    """Converts text to speech using gTTS, speeds it up, and saves it."""
    output_audio = "response.mp3"
    
    # Generate speech with gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(output_audio)

    # Load and speed up audio using Pydub
    audio = AudioSegment.from_file(output_audio)
    faster_audio = audio.speedup(playback_speed=1.35)  # 30% faster
    
    # Save the faster version
    output_fast_audio = "response_fast.mp3"
    faster_audio.export(output_fast_audio, format="mp3")

    return output_fast_audio

def generate_speech_google(text: str) -> str:
    """Uses Google Cloud TTS to generate custom voice speech."""
    client = texttospeech.TextToSpeechClient()
    
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-F",  # Change voice type here
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    
    output_audio = "response_google.mp3"
    with open(output_audio, "wb") as out:
        out.write(response.audio_content)

    return output_audio
