# 🔮 Alexa Voice Assistant

An AI-powered voice assistant built with **Streamlit**, **OpenAI Whisper**, **Groq LLaMA**, and **Tavily Web Search** — capable of understanding speech, detecting intent, and responding intelligently in real time.

---

## ✨ Features

- 🎙️ **Speech-to-Text** — Records your voice and transcribes it using OpenAI Whisper
- 🧠 **Intent Classification** — Deep learning model detects what you want (Q&A, music, tasks, weather, etc.)
- 🤖 **AI Responses** — Groq LLaMA answers general questions with short, natural replies
- 🌐 **Live Web Search** — Tavily fetches real-time answers for current events and facts
- 📋 **To-Do Tasks** — Add tasks by voice command
- 🎵 **Play Music** — Opens YouTube with your requested song
- 💬 **Conversation History** — Tracks all interactions with intent confidence scores
- 🖥️ **Streamlit UI** — Clean, real-time web dashboard

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web UI |
| OpenAI Whisper | Speech-to-text transcription |
| Groq LLaMA 3 | AI language model for responses |
| Tavily API | Real-time web search |
| scikit-learn | Intent classification model |
| sounddevice | Audio recording |
| FFmpeg | Audio processing (install separately) |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Amshavarthana-S/alexa-voice-assistant.git
cd alexa-voice-assistant
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg
Download from: https://www.gyan.dev/ffmpeg/builds/

Extract and add the `bin/` folder to your system PATH.

### 4. Set up API Keys

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys here:
- Groq: https://console.groq.com/keys
- Tavily: https://app.tavily.com

---

## ▶️ Running the App

```bash
streamlit run alexa.py
```

Open your browser at `http://localhost:8501`

---

## 🗣️ Example Voice Commands

| You Say | Alexa Does |
|---------|-----------|
| "What is quantum physics?" | Answers via Groq/Tavily |
| "Play Spotify songs" | Opens YouTube |
| "Add buy groceries to my list" | Adds to task list |
| "What's the weather today?" | Opens weather forecast |
| "Who is the Prime Minister of India?" | Live web search answer |

---

## 📁 Project Structure

```
alexa-voice-assistant/
├── alexa.py                    # Main Streamlit app
├── alexa_data.csv              # Training data for intent classifier
├── alexa_data_clean.csv        # Cleaned training data
├── Alexa_Voice_Assistant.ipynb # Jupyter notebook (exploration)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignored files
└── .env                        # API keys (never commit this!)
```

---

## 🔒 Security Notes

- Never commit your `.env` file
- Regenerate API keys if accidentally exposed
- `.env` is listed in `.gitignore` for safety

---

## 📦 Requirements

See `requirements.txt` for full list. Key packages:
- `streamlit`
- `openai-whisper`
- `groq`
- `tavily-python`
- `sounddevice`
- `scikit-learn`
- `python-dotenv`

---

Built with ❤️ by [Amshavarthana-S](https://github.com/Amshavarthana-S)
