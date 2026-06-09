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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

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
DATA_PATH  = r"C:\Users\Amsha\Desktop\AlexaVoiceAssistant\alexa_data.csv"

sd.default.device = (1, 5)

# Session States
if "last_input"      not in st.session_state: st.session_state.last_input      = "None"
if "last_output"     not in st.session_state: st.session_state.last_output     = "System ready. Click the button and speak!"
if "last_intent"     not in st.session_state: st.session_state.last_intent     = "N/A"
if "last_confidence" not in st.session_state: st.session_state.last_confidence = 0.0
if "is_recording"    not in st.session_state: st.session_state.is_recording    = False
if "history" not in st.session_state: st.session_state.history = []

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
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["prompt"])
    intent_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    intent_model.fit(X, df["intent"])
    return vectorizer, intent_model
vectorizer, intent_model = load_intent_model()

# -------------------------------
# Core Functions
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
    if not text: return "unknown", 0.0
    X_test = vectorizer.transform([text])
    intent = intent_model.predict(X_test)[0]
    proba = intent_model.predict_proba(X_test)[0]
    return intent, round(max(proba) * 100, 2)

def speak(text):
    clean_text = text.replace("'", "").replace('"', '')
    subprocess.Popen([
        'PowerShell', '-Command',
        f'Add-Type -AssemblyName System.Speech; '
        f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        f'$s.Speak("{clean_text}")'
    ])

def perform_action(intent, prompt=""):
    if intent == "alarm":
        return "Setting your alarm!"
    elif intent == "play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Playing music on YouTube!"
    elif intent == "open_website":
        if "youtube" in prompt:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube!"
        elif "google" in prompt:
            webbrowser.open("https://www.google.com")
            return "Opening Google!"
        elif "github" in prompt:
            webbrowser.open("https://www.github.com")
            return "Opening GitHub!"
        return "Which website should I open?"
    elif intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today's top news headlines!"
    elif intent == "date_time":
        return datetime.datetime.now().strftime("It is %I:%M %p on %B %d, %Y")
    elif intent == "jokes_fun":
        return random.choice([
            "Why do programmers wear glasses? Because they can't C#.",
            "Why did the computer go to the doctor? It caught a virus!",
            "I would tell you a UDP joke, but you might not get it."
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
            "Octopuses have three hearts and blue blood."
        ])
    elif intent == "dictionary":
        word = prompt.replace("define", "").replace("meaning of", "").strip()
        webbrowser.open(f"https://www.merriam-webster.com/dictionary/{word}")
        return f"Looking up the meaning of: {word}!"
    elif intent == "shopping_list":
        item = re.sub(r"(add|buy|put|shopping list|on list)", "", prompt).strip()
        return f'Added "{item}" to your shopping list!'
    elif intent == "translation":
        webbrowser.open(f"https://translate.google.com/?text={prompt.replace(' ', '+')}")
        return "Opening Google Translate!"
    elif intent == "directions":
        webbrowser.open(f"https://www.google.com/maps/search/{prompt.replace(' ', '+')}")
        return "Opening Google Maps!"
    elif intent == "movies":
        webbrowser.open("https://www.imdb.com/chart/popular")
        return "Opening popular movies on IMDB!"
    elif intent == "games_trivia":
        return random.choice([
            "Trivia: The Great Wall of China is not visible from space!",
            "Trivia: A group of flamingos is called a flamboyance!",
            "Trivia: The shortest war in history lasted only 38 minutes!"
        ])
    elif intent == "personality":
        return random.choice([
            "I am Alexa, your AI voice assistant!",
            "I am built with Whisper and Deep Learning!",
            "I can help you with music, news, weather, jokes and much more!"
        ])
    elif intent == "set_timer":
        return "Timer started!"
    elif intent == "reminder":
        return "Reminder set!"
    elif intent == "smart_home":
        return "Smart home command received!"
    else:
        return "Sorry, I did not understand that. Please try again!"

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

    st.markdown(f"""
    <div class="glass-card">
        <h4 style="margin-top:0; color:#8b949e;">LAST RESULT</h4>
        <p style="margin-bottom:5px;">Detected Intent:</p>
        <div class="intent-pill">{st.session_state.last_intent.upper()}</div>
        <br><br>
        <p style="margin-bottom:5px;">Confidence Score:</p>
        <h2 style="margin-top:0; color:#4facfe;">{st.session_state.last_confidence}%</h2>
        <hr style="border-color:#30363d;">
        <span style="font-size:0.8rem; color:#8b949e;"><b>Mic:</b> Device {sd.default.device[0]} | <b>Speaker:</b> Device {sd.default.device[1]}</span>
    </div>
    """, unsafe_allow_html=True)

    duration = st.slider("Recording Duration (seconds)", 3, 10, 5)

# RIGHT PANEL
with dashboard_right:
    st.markdown("### 🖥️ Response Window")

    exec_trigger = st.button("🎤 Click to Speak")

    if exec_trigger:
        st.session_state.is_recording = True
        st.rerun()

    if st.session_state.is_recording:
        record_audio(duration=duration)
        st.session_state.is_recording = False

        user_raw_string = speech_to_text()
        if user_raw_string.strip():
            st.session_state.last_input = user_raw_string

            intent_pred, confidence_score = predict_intent(user_raw_string)
            st.session_state.last_intent = intent_pred
            st.session_state.last_confidence = confidence_score

            output_response = perform_action(intent_pred, user_raw_string)
            st.session_state.last_output = output_response

            speak(output_response)
            st.session_state.history.append({
                "time": datetime.datetime.now().strftime("%I:%M %p"),
                "you": user_raw_string,
                "intent": intent_pred,
                "confidence": confidence_score,
                "alexa": output_response
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
        """)
    with col_m2:
        st.markdown("""
        - 🌐 *"open youtube"*
        - 📖 *"define gravity"*
        - 🗺️ *"navigate to hospital"*
        - 🛒 *"add milk to shopping list"*
        - 🎮 *"play a game"*
        - 🤖 *"who are you"*
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
        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid #30363d;">
            <span style="color:#8b949e; font-size:0.8rem;">🕐 {entry['time']}</span>
            <p style="margin:5px 0; color:#f0f6fc;">🗣️ <b>You:</b> {entry['you']}</p>
            <p style="margin:5px 0; color:#8b949e;">🧠 <b>Intent:</b> <span style="color:#00f2fe;">{entry['intent']}</span> — <b>Confidence:</b> {entry['confidence']}%</p>
            <p style="margin:5px 0; color:#4facfe;">🤖 <b>Alexa:</b> {entry['alexa']}</p>
        </div>
        """, unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit + OpenAI Whisper + Deep Learning")