#  Alexa Voice Assistant

An AI-powered voice assistant built from scratch using OpenAI Whisper, Deep Learning intent classification, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-orange)

---

##  Architecture
Voice Input → Whisper STT → TF-IDF → MLP Classifier → Action Engine → TTS Output


---

##  Features

- 🎙️ Real-time voice recording via microphone
- 🤖 Speech to text using OpenAI Whisper
- 🧠 Deep Learning intent detection (MLP Neural Network)
- 📊 Confidence score display
- ⚙️ 20+ supported voice commands
- 💬 Conversation history
- 🔊 Text to speech response
- 🔮 Premium dark glassmorphic UI

---

##  Supported Commands

| Say This | Action |
|---|---|
| "play music for me" | Opens YouTube music |
| "what time is it" | Tells current time |
| "tell me a joke" | Tells a joke |
| "latest news" | Opens Google News |
| "weather today" | Opens weather site |
| "open youtube" | Opens YouTube |
| "open google" | Opens Google |
| "define gravity" | Opens dictionary |
| "navigate to hospital" | Opens Google Maps |
| "add milk to shopping list" | Adds to shopping list |
| "random fact" | Tells a fun fact |
| "who are you" | Assistant introduces itself |
| "play a game" | Gives trivia question |
| "translate hello to spanish" | Opens Google Translate |
| "open calculator" | Opens calculator |

---

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web UI |
| OpenAI Whisper | Speech to Text |
| scikit-learn MLP | Intent classification |
| TF-IDF Vectorizer | Text to numbers |
| SoundDevice | Mic recording |
| FFmpeg | Audio decoding |
| PowerShell TTS | Text to Speech |

---

##  How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Amshavarthana-S/alexa-voice-assistant
cd alexa-voice-assistant
2. Install dependencies

pip install -r requirements.txt
3. Install FFmpeg

Download from gyan.dev and add to PATH.

4. Update paths in alexa.py

Change these lines to match your system:

ffmpeg_path = r"YOUR_FFMPEG_PATH\bin"
AUDIO_PATH  = r"YOUR_PATH\input.wav"
DATA_PATH   = r"YOUR_PATH\alexa_data.csv"
5. Run the app

streamlit run alexa.py
📁 Project Structure
alexa-voice-assistant/
├── alexa.py                      # Main Streamlit app
├── alexa_data.csv                # Training data (20+ intents)
├── Alexa_Voice_Assistant.ipynb   # Development notebook
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
UI Preview
Dark glassmorphic dashboard with neon ring animation, confidence score, and conversation history.

Author
Amshavarthana S
🔗 GitHub
Built with using Python, Streamlit and OpenAI Whisper


