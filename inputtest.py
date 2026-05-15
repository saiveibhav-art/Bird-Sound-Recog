import pyaudio
import wave
import numpy as np
import os
from datetime import datetime
from scipy.signal import butter, lfilter
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# --- CONFIGURATION ---
INDEX = 1
RATE = 44100  # Recommended to keep at 44.1kHz for standard hardware compatibility
CHANNELS = 2
RECORD_SECONDS = 15 
TEMP_FILENAME = "temp_capture.wav" # Temporary file for AI to read

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data, cutoff=1000, fs=44100):
    """Refined High-Pass Filter to remove laptop fan noise"""
    b, a = butter_highpass(cutoff, fs, order=5)
    y = lfilter(b, a, data)
    return y.astype(np.int16)

def record_and_analyze(analyzer):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE,
                        input=True, input_device_index=INDEX, frames_per_buffer=8192)
        
        print(f"\n[LISTENING] 15s window active...")
        frames = []
        for _ in range(0, int(RATE / 8192 * RECORD_SECONDS)):
            data = stream.read(8192, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
        
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()

    # --- SIGNAL PROCESSING ---
    full_signal = np.concatenate(frames)
    filtered_signal = apply_filter(full_signal, cutoff=1000, fs=RATE)
    
    # Normalization for cleaner AI read
    max_val = np.max(np.abs(filtered_signal))
    if max_val > 0:
        normalized_signal = (filtered_signal / max_val * 32767).astype(np.int16)
    else:
        normalized_signal = filtered_signal

    # Save to temporary file for BirdNET to analyze
    with wave.open(TEMP_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(normalized_signal.tobytes())

    # --- AI ANALYSIS ---
    recording = Recording(analyzer, TEMP_FILENAME, lat=12.97, lon=77.59, min_conf=0.35)
    recording.analyze()
    
    if recording.detections:
        # Get the highest confidence detection for the filename
        best_detection = recording.detections[0]
        bird_name = best_detection['common_name'].replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique filename: e.g., "Common_Myna_20260513_134500.wav"
        new_filename = f"{bird_name}_{timestamp}.wav"
        
        # Save the finalized recording with the new name
        with wave.open(new_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(normalized_signal.tobytes())
            
        print(f"✅ DETECTED: {best_detection['common_name']} ({best_detection['confidence']*100:.1f}%)")
        print(f"💾 SAVED AS: {new_filename}")
    else:
        print("❌ No detection. Deleting temporary audio...")
        if os.path.exists(TEMP_FILENAME):
            os.remove(TEMP_FILENAME)

if __name__ == "__main__":
    print("Loading BirdNET Analyzer...")
    analyzer = Analyzer()
    try:
        while True:
            record_and_analyze(analyzer)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")