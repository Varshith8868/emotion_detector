# audio.py
import os
import threading

# Try to import playsound and pyttsx3
try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)  # speaking speed
except Exception:
    _tts_engine = None

SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")

def _play_file(path):
    try:
        if playsound:
            playsound(path)
        else:
            # fallback to pyttsx3 speak if playsound missing
            if _tts_engine:
                _tts_engine.say(os.path.splitext(os.path.basename(path))[0])
                _tts_engine.runAndWait()
    except Exception:
        pass

def speak_emotion(emotion):
    """
    Play a recorded file if available in sounds/<emotion>.mp3
    otherwise use TTS fallback to speak the emotion word.
    """
    fname = f"{emotion.lower()}.mp3"
    fpath = os.path.join(SOUNDS_DIR, fname)
    if os.path.isfile(fpath):
        threading.Thread(target=_play_file, args=(fpath,), daemon=True).start()
    else:
        # TTS fallback
        if _tts_engine:
            threading.Thread(target=lambda: (_tts_engine.say(emotion), _tts_engine.runAndWait()), daemon=True).start()
