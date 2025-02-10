from gtts import gTTS
import os

text = "Hello, how can I help you?"
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")

# Play the audio (Linux/macOS)
os.system("mpg321 output.mp3")

# Play the audio (Windows)
os.system("start output.mp3")
