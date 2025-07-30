import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile
import requests
import os
import queue
import threading
import shutil
from plyer import notification
from playsound import playsound
from langchain_community.llms import Ollama
from scipy.signal import butter, lfilter


model = whisper.load_model("large")
llm = Ollama(model="mistral")


fs = 44100
chunk_duration = 5  # seconds
chunk_samples = int(fs * chunk_duration)
buffer = np.empty((0, 1), dtype='int16')
audio_q = queue.Queue()
history_audio = []

# 🎚️ Normalize & noise reduction
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))#max value of audio
    return (audio / max_val * 32767).astype(np.int16) if max_val > 0 else audio#normalize audio

def reduce_noise(audio):
    # Low-pass filter example
    b, a = butter(6, 0.15, btype='low')
    filtered = lfilter(b, a, audio.flatten())
    return filtered.reshape(-1, 1).astype(np.int16)


import pygame

def alert_on_laptop():
    try:
        notification.notify(
            title="📢 Your name was mentioned",
            message="Someone is calling you in the meeting!",
            timeout=5
        )
        
        sound_path = os.path.join(os.path.dirname(__file__), "alert.mp3")

        if os.path.exists(sound_path):
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            
            
            while pygame.mixer.music.get_busy():
                continue
        else:
            print("⚠️ alert.mp3 file not found.")

    except Exception as e:
        print("❌ Failed to play alert:", e)




def notify_n8n_webhook(message="Your name was mentioned in the meeting"):
    try:
        requests.post(
            "https://ahmedfawzi702.app.n8n.cloud/webhook/9d8c9bb8-0a1c-47ea-8921-e005afb72e54",
            json={"text": message}
        )
    except Exception as e:
        print("❌ Failed to notify webhook:", e)


def check_name_in_text(text, name="أحمد فوزي"):
    prompt = f"""
النص التالي من ميتنج:
"{text}"

هل هذا الحديث موجه إلى {name} أو تم ذكر اسمه؟ جاوب بـ نعم أو لا فقط.
"""
    try:
        response = llm.invoke(prompt)
        return "نعم" in response.lower()
    except Exception as e:
        print("❌ LLM error:", e)
        return False


def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️ Mic status:", status)
    audio_q.put(indata.copy())

# 💾 Save wav
def save_audio_segment(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        scipy.io.wavfile.write(f.name, fs, data)
        return f.name


def process_audio():
    global buffer, history_audio
    while True:
        try:
            data = audio_q.get()
            buffer = np.append(buffer, data, axis=0)
            history_audio.append(data)

            if len(history_audio) > 6:
                history_audio.pop(0)

            if len(buffer) >= chunk_samples:
                segment = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                # 🔊 Clean audio
                clean = normalize_audio(reduce_noise(segment))
                audio_path = save_audio_segment(clean)

                try:
                    result = model.transcribe(audio_path, language="ar")
                    text = result["text"].strip()
                except Exception as e:
                    print("❌ Transcription error:", e)
                    text = ""
                os.remove(audio_path)

                if len(text) <= 3:
                    print("⏭️ Ignored short text.")
                    continue

                print(f"📝 {text}")

                if check_name_in_text(text):
                    print("✅ Your name was mentioned!")

                    context_parts = history_audio[-2:] + [data]
                    for _ in range(2):
                        try:
                            context_parts.append(audio_q.get(timeout=2))
                        except queue.Empty:
                            break

                    full_audio = np.concatenate(context_parts, axis=0)
                    context_clean = normalize_audio(reduce_noise(full_audio))
                    context_path = save_audio_segment(context_clean)

                    try:
                        context_result = model.transcribe(context_path, language="ar")
                        context_text = context_result["text"].strip()
                        print("🗣️ Context around name mention:\n")
                        print("🔸", context_text)
                    except Exception as e:
                        print("⚠️ Failed to get context:", e)
                    finally:
                        os.remove(context_path)

                    alert_on_laptop()
                    notify_n8n_webhook()
                else:
                    print("❌ No name mention detected.")

        except Exception as e:
            print("❌ Processing error:", e)

# ▶️ Main loop
def main():
    print("🎧 Listening... Press Ctrl+C to stop.")
    try:
        threading.Thread(target=process_audio, daemon=True).start()
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=audio_callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    main()
