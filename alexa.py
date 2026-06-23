import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample
import whisper
import pandas as pd
import datetime
import webbrowser
import random
import re
import os
import numpy as np
import subprocess
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from groq import Groq

# -------------------------------
# Streamlit Dashboard Configuration
# -------------------------------
st.set_page_config(
    page_title="Alexa Ambient Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main .block-container { max-width: 1200px; padding-top: 1.5rem; background-color: #0d1117; }
    body, [data-testid="stAppViewContainer"] { background-color: #0d1117; color: #f0f6fc; }

    .echo-container { display: flex; justify-content: center; align-items: center; padding: 20px; }
    .echo-ring-idle {
        width: 80px; height: 80px; border-radius: 50%;
        background: transparent; border: 4px solid #00f2fe;
        box-shadow: 0 0 20px #00f2fe, inset 0 0 20px #00f2fe;
    }
    .echo-ring-recording {
        width: 80px; height: 80px; border-radius: 50%;
        background: transparent; border: 4px solid #ff0844;
        box-shadow: 0 0 25px #ff0844, inset 0 0 25px #ff0844;
        animation: pulse-ring 1s infinite alternate;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.95); opacity: 0.8; }
        100% { transform: scale(1.05); opacity: 1; }
    }

    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .intent-pill {
        background: linear-gradient(135deg, #1f4068, #162447);
        border-radius: 20px; padding: 6px 16px;
        font-weight: 600; color: #00f2fe; display: inline-block;
    }
    .error-pill {
        background: linear-gradient(135deg, #4a1c1c, #2d0a0a);
        border-radius: 20px; padding: 6px 16px;
        font-weight: 600; color: #ff6b6b; display: inline-block;
        font-size: 0.75rem;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #0d1117 !important; border: none; font-weight: bold;
        border-radius: 12px; height: 50px; font-size: 1.1rem;
        transition: all 0.3s ease; width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 242, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Paths & Config
# -------------------------------
ffmpeg_path = r"C:\Users\Amsha\Desktop\AlexaVoiceAssistant\ffmpeg\ffmpeg-8.1.1-essentials_build\bin"
os.environ["PATH"] = os.environ["PATH"] + os.pathsep + ffmpeg_path

AUDIO_PATH = r"C:\Users\Amsha\Desktop\AlexaVoiceAssistant\input.wav"
DATA_PATH  = r"C:\Users\Amsha\Desktop\AlexaVoiceAssistant\alexa_data_clean.csv"

CONFIDENCE_THRESHOLD = 40

sd.default.device = (1, 5)

# -------------------------------
# API Keys
# -------------------------------
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# Tavily Web Search
# Fetches real-time info from the web — fixes outdated/wrong answers
# -------------------------------
def search_tavily(query):
    """
    Calls Tavily Search API for real-time web results.
    Returns a clean answer string or None on failure.
    """
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": True
        }
        response = requests.post(url, json=payload, timeout=8)
        data = response.json()

        # Tavily returns a direct answer string when include_answer=True
        if data.get("answer"):
            return data["answer"].strip()

        # Fallback: use top result snippet
        results = data.get("results", [])
        if results:
            return results[0].get("content", "")[:300].strip()

        return None
    except Exception as e:
        st.session_state.last_groq_error = f"Tavily error: {str(e)}"
        return None


# -------------------------------
# Groq LLaMA — fallback when Tavily has no answer
# -------------------------------
def ask_groq(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Alexa, a helpful voice assistant. "
                        "Give short, clear answers in 1-2 sentences only. "
                        "No bullet points. Speak naturally like a voice assistant."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()
        return result if result else "I got an empty response. Please try rephrasing."
    except Exception as e:
        error_msg = str(e)
        st.session_state.last_groq_error = error_msg
        return f"Groq error: {error_msg}"


# -------------------------------
# Smart Answer: Tavily first, Groq as fallback
# Used for ALL factual / Q&A / unknown queries
# -------------------------------
def smart_answer(prompt):
    """
    Pipeline:
    1. Tavily (real-time web) → accurate for current events, politics, people
    2. Groq LLaMA → fallback for anything Tavily misses
    """
    answer = search_tavily(prompt)
    if answer:
        return answer
    return ask_groq(prompt)


# -------------------------------
# Session States
# -------------------------------
if "last_input"       not in st.session_state: st.session_state.last_input       = "None"
if "last_output"      not in st.session_state: st.session_state.last_output      = "System ready. Click the button and speak!"
if "last_intent"      not in st.session_state: st.session_state.last_intent      = "N/A"
if "last_confidence"  not in st.session_state: st.session_state.last_confidence  = 0.0
if "is_recording"     not in st.session_state: st.session_state.is_recording     = False
if "history"          not in st.session_state: st.session_state.history          = []
if "last_groq_error"  not in st.session_state: st.session_state.last_groq_error  = ""

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")
model = load_whisper()

@st.cache_resource
def load_intent_model():
    df = pd.read_csv(DATA_PATH)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["prompt"])
    intent_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    intent_model.fit(X, df["intent"])
    return vectorizer, intent_model
vectorizer, intent_model = load_intent_model()

# -------------------------------
# Core Audio Functions
# -------------------------------
def record_audio(duration=5, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    recording = recording / (np.max(np.abs(recording)) + 1e-9)
    num_samples = int(len(recording) * 16000 / fs)
    resampled = resample(recording, num_samples)
    write(AUDIO_PATH, 16000, resampled)

def speech_to_text():
    result = model.transcribe(AUDIO_PATH, language="en", temperature=0.0)
    return result["text"].lower().strip()

def predict_intent(text):
    if not text:
        return "unknown", 0.0
    X_test = vectorizer.transform([text])
    intent = intent_model.predict(X_test)[0]
    proba = intent_model.predict_proba(X_test)[0]
    confidence = round(max(proba) * 100, 2)
    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return intent, confidence

def speak(text):
    clean_text = text.replace("'", "").replace('"', '')
    subprocess.Popen([
        'PowerShell', '-Command',
        f'Add-Type -AssemblyName System.Speech; '
        f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        f'$s.Speak("{clean_text}")'
    ])

# -------------------------------
# Perform Action
# -------------------------------
def perform_action(intent, prompt=""):

    # Unknown intent → real-time smart answer
    if intent == "unknown":
        return smart_answer(prompt)

    # General Q&A → Tavily web search first, Groq fallback
    # Replaces Wikipedia which gave wrong/outdated answers
    elif intent == "general_qa":
        return smart_answer(prompt)

    elif intent == "alarm":
        return "Setting your alarm!"

    elif intent == "play_music":
        query = re.sub(r"(play|music|song|songs|start|put on)", "", prompt).strip()
        search = query if query else "music"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search.replace(' ', '+')}")
        return f"Playing {search} on YouTube!"

    elif intent == "open_website":
        sites = {
            "youtube":   "https://www.youtube.com",
            "google":    "https://www.google.com",
            "github":    "https://www.github.com",
            "facebook":  "https://www.facebook.com",
            "instagram": "https://www.instagram.com",
            "twitter":   "https://www.twitter.com",
            "netflix":   "https://www.netflix.com",
            "reddit":    "https://www.reddit.com",
        }
        for site, url in sites.items():
            if site in prompt:
                webbrowser.open(url)
                return f"Opening {site.capitalize()}!"
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return "Which website should I open? Try saying 'open YouTube' or 'open Google'."

    elif intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today's top news headlines!"

    elif intent == "date_time":
        return datetime.datetime.now().strftime("It is %I:%M %p on %B %d, %Y")

    elif intent == "jokes_fun":
        return random.choice([
            "Why do programmers wear glasses? Because they can't C#.",
            "Why did the computer go to the doctor? It caught a virus!",
            "I would tell you a UDP joke, but you might not get it.",
            "Why do Java developers wear glasses? Because they don't C#.",
            "A SQL query walks into a bar, walks up to two tables and asks: Can I join you?",
            "Why was the JavaScript developer sad? Because he didn't know how to null his feelings.",
        ])

    elif intent == "weather":
        webbrowser.open("https://www.weather.com")
        return "Opening weather forecast!"

    elif intent == "calculator":
        os.system("calc")
        return "Opening calculator!"

    elif intent == "facts":
        return random.choice([
            "Did you know: Honey never spoils. Archaeologists found 3000-year-old honey in Egyptian tombs!",
            "A day on Venus is longer than a year on Venus.",
            "Octopuses have three hearts and blue blood.",
            "Bananas are berries, but strawberries are not!",
            "A bolt of lightning is five times hotter than the surface of the Sun.",
        ])

    elif intent == "dictionary":
        word = re.sub(r"(define|meaning of|what is|definition of)", "", prompt).strip()
        webbrowser.open(f"https://www.merriam-webster.com/dictionary/{word}")
        return f"Looking up the meaning of: {word}!"

    elif intent == "shopping_list":
        item = re.sub(r"(add|buy|put|shopping list|on list|to list)", "", prompt).strip()
        return f'Added "{item}" to your shopping list!'

    elif intent == "translation":
        webbrowser.open(f"https://translate.google.com/?text={prompt.replace(' ', '+')}")
        return "Opening Google Translate!"

    elif intent == "directions":
        destination = re.sub(r"(navigate to|directions to|way to|route to|take me to|how to get to)", "", prompt).strip()
        webbrowser.open(f"https://www.google.com/maps/search/{destination.replace(' ', '+')}")
        return f"Opening Google Maps for {destination}!"

    elif intent == "movies":
        query = re.sub(r"(suggest|recommend|show|movie|film|watch)", "", prompt).strip()
        if query:
            webbrowser.open(f"https://www.imdb.com/search/title/?title={query.replace(' ', '+')}")
            return f"Searching for {query} on IMDB!"
        webbrowser.open("https://www.imdb.com/chart/popular")
        return "Opening popular movies on IMDB!"

    elif intent == "games_trivia":
        return random.choice([
            "Trivia: The Great Wall of China is not visible from space!",
            "Trivia: A group of flamingos is called a flamboyance!",
            "Trivia: The shortest war in history lasted only 38 minutes!",
            "Trivia: Cleopatra lived closer in time to the Moon landing than to the construction of the Great Pyramid!",
        ])

    elif intent == "personality":
        return random.choice([
            "I am Alexa, your AI voice assistant!",
            "I am built with Whisper and Deep Learning!",
            "I can help you with music, news, weather, jokes and much more!",
            "I was trained using TF-IDF and an MLP neural network!",
        ])

    elif intent == "set_timer":
        return "Timer started! I'll let you know when it's done."

    elif intent == "reminder":
        item = re.sub(r"(remind me|reminder|don't forget|remember)", "", prompt).strip()
        return f'Reminder set for: "{item}"!'

    elif intent == "smart_home":
        return "Smart home command received!"

    elif intent == "calendar":
        return "Opening your calendar!"

    elif intent == "traffic":
        webbrowser.open("https://www.google.com/maps")
        return "Checking traffic conditions on Google Maps!"

    elif intent == "routine":
        return "Starting your routine!"

    elif intent == "todo_task":
        task = re.sub(r"(add task|new task|to do|todo)", "", prompt).strip()
        return f'Task added: "{task}"!'

    elif intent == "unit_conversion":
        webbrowser.open(f"https://www.google.com/search?q={prompt.replace(' ', '+')}")
        return f"Looking up conversion for: {prompt}"

    elif intent == "podcast_audiobook":
        webbrowser.open("https://open.spotify.com/")
        return "Opening Spotify for podcasts and audiobooks!"

    elif intent == "device_status":
        return "Checking your device status!"

    elif intent == "play_radio":
        webbrowser.open("https://www.radio.net")
        return "Opening Radio.net!"

    else:
        return smart_answer(prompt)


# -------------------------------
# Dashboard Header
# -------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<h1 style='margin-bottom:0; font-weight:800; letter-spacing:-1px;'>🔮 AMBIENT INTELLIGENCE HUB</h1>", unsafe_allow_html=True)
    st.caption("Real-Time Voice Analysis Engine")
with col_h2:
    st.markdown("<div style='text-align:right; margin-top:15px; color:#8b949e;'>System Active • 2026</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Main Layout
# -------------------------------
dashboard_left, dashboard_right = st.columns([1, 1.8], gap="large")

# LEFT PANEL
with dashboard_left:
    st.markdown("### 📊 System Info")

    if st.session_state.is_recording:
        st.markdown('<div class="echo-container"><div class="echo-ring-recording"></div></div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#ff0844; font-weight:bold;'>LISTENING...</p>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="echo-container"><div class="echo-ring-idle"></div></div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#00f2fe;'>READY</p>", unsafe_allow_html=True)

    groq_error_html = ""
    if st.session_state.last_groq_error:
        short_err = st.session_state.last_groq_error[:80]
        groq_error_html = f"""
        <br>
        <p style="margin-bottom:5px; color:#8b949e;">Last Error:</p>
        <div class="error-pill">{short_err}...</div>
        """

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="margin-top:0; color:#8b949e;">LAST RESULT</h4>
        <p style="margin-bottom:5px;">Detected Intent:</p>
        <div class="intent-pill">{st.session_state.last_intent.upper()}</div>
        <br><br>
        <p style="margin-bottom:5px;">Confidence Score:</p>
        <h2 style="margin-top:0; color:#4facfe;">{st.session_state.last_confidence}%</h2>
        <hr style="border-color:#30363d;">
        <span style="font-size:0.8rem; color:#8b949e;">
            <b>Mic:</b> Device {sd.default.device[0]} |
            <b>Speaker:</b> Device {sd.default.device[1]}
        </span>
        {groq_error_html}
    </div>
    """, unsafe_allow_html=True)

    duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

# RIGHT PANEL
with dashboard_right:
    st.markdown("### 🖥️ Response Window")

    exec_trigger = st.button("🎤 Click to Speak")

    if exec_trigger:
        st.session_state.is_recording = True
        st.session_state.last_groq_error = ""
        st.rerun()

    if st.session_state.is_recording:
        record_audio(duration=duration)
        st.session_state.is_recording = False

        user_raw_string = speech_to_text()
        if user_raw_string.strip():
            st.session_state.last_input = user_raw_string

            intent_pred, confidence_score = predict_intent(user_raw_string)
            st.session_state.last_intent     = intent_pred
            st.session_state.last_confidence = confidence_score

            output_response = perform_action(intent_pred, user_raw_string)
            st.session_state.last_output = output_response

            speak(output_response)

            via_web = intent_pred in ("unknown", "general_qa")
            st.session_state.history.append({
                "time":       datetime.datetime.now().strftime("%I:%M %p"),
                "you":        user_raw_string,
                "intent":     intent_pred,
                "confidence": confidence_score,
                "alexa":      output_response,
                "via_web":    via_web
            })
        else:
            st.session_state.last_output = "Nothing heard. Please speak louder and try again!"
        st.rerun()

    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #00f2fe;">
        <h5 style="margin-top:0; color:#8b949e;">YOU SAID</h5>
        <p style="font-size:1.25rem; font-style:italic; color:#f0f6fc;">"{st.session_state.last_input}"</p>
    </div>

    <div class="glass-card" style="border-left: 4px solid #4facfe; background: linear-gradient(to right, rgba(22,27,34,0.9), rgba(13,17,23,0.9));">
        <h5 style="margin-top:0; color:#8b949e;">ALEXA SAYS</h5>
        <p style="font-size:1.35rem; color:#00f2fe; font-weight:500;">{st.session_state.last_output}</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Commands Reference
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 What Can You Say?"):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        - 🎵 *"play music for me"*
        - 🕐 *"what time is it"*
        - 😂 *"tell me a joke"*
        - 📰 *"latest news"*
        - 🌤️ *"weather today"*
        - 🧮 *"open calculator"*
        - 🤔 *"who is Elon Musk"*
        - 🤔 *"what is machine learning"*
        """)
    with col_m2:
        st.markdown("""
        - 🌐 *"open youtube"*
        - 📖 *"define gravity"*
        - 🗺️ *"navigate to hospital"*
        - 🛒 *"add milk to shopping list"*
        - 🎮 *"quiz time"*
        - 🤖 *"who are you"*
        - 🎵 *"play tamil songs"*
        - 📻 *"start radio"*
        """)

# -------------------------------
# Conversation History
# -------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Conversation History")

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

    for entry in reversed(st.session_state.history):
        web_tag = " 🌐 <span style='color:#3fb950; font-size:0.75rem;'>via Web</span>" if entry.get("via_web") else ""
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #30363d;">
            <span style="color:#8b949e; font-size:0.8rem;">🕐 {entry['time']}</span>
            <p style="margin:5px 0; color:#f0f6fc;">🗣️ <b>You:</b> {entry['you']}</p>
            <p style="margin:5px 0; color:#8b949e;">🧠 <b>Intent:</b>
                <span style="color:#00f2fe;">{entry['intent']}</span>
                — <b>Confidence:</b> {entry['confidence']}%{web_tag}
            </p>
            <p style="margin:5px 0; color:#4facfe;">🤖 <b>Alexa:</b> {entry['alexa']}</p>
        </div>
        """, unsafe_allow_html=True)

st.caption("Built with ❤️ using Streamlit + OpenAI Whisper + Deep Learning + Groq LLaMA + Tavily Web Search")
