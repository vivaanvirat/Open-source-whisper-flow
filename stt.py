import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pynput import keyboard
from pynput.keyboard import Controller, Key
import tempfile
import os
import threading
import time

class VoiceTyper:
    def __init__(self):
        models = ['tiny','base','small','medium','large', 'turbo']
        for i in range(6):
            print(f'{i+1} {models[i]}')
        size = input('Choose a model(Type number 1-6): ')
        print("Loading Whisper model... (this may take a moment)")
        self.model = whisper.load_model(models[size-1])
        self.keyboard_controller = Controller()
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.recording_thread = None
        self.alt_pressed = False
        self.s_pressed = False
        print("Whisper model loaded successfully!")
        print("=" * 50)
        print("INSTRUCTIONS:")
        print("1. Press and HOLD Alt+S to start recording")
        print("2. Speak while holding both keys")
        print("3. Release to stop and transcribe")
        print("4. Press Ctrl+C to exit")
        print("=" * 50)
        print("\nReady! Waiting for Alt+S...\n")
    
    def record_audio(self):
        """Record audio while Alt+S is held"""
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Recording status: {status}")
            if self.is_recording:
                self.audio_data.append(indata.copy())
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, 
                               callback=callback, dtype='float32'):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Recording error: {e}")
    
    def transcribe_and_type(self):
        """Transcribe the recorded audio and type it out"""
        if not self.audio_data or len(self.audio_data) == 0:
            print("❌ No audio recorded (recording was too short or microphone issue)")
            return
        
        print("🎤 Processing audio...")
        
        try:
            # Convert audio data to numpy array
            audio_array = np.concatenate(self.audio_data, axis=0)
            audio_array = audio_array.flatten()
            
            # Check if we have actual audio data
            if len(audio_array) < self.sample_rate * 0.1:  # Less than 0.1 seconds
                print("❌ Recording too short, try holding Alt+S longer")
                return
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
                wav.write(temp_filename, self.sample_rate, (audio_array * 32767).astype(np.int16))
            
            # Transcribe using Whisper
            print("🔄 Transcribing...")
            result = self.model.transcribe(temp_filename, language='en', fp16=False)
            transcribed_text = result['text'].strip()
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            if transcribed_text:
                print(f"✅ Transcribed: '{transcribed_text}'")
                print("⌨️  Typing...")
                # Small delay to ensure the keys are released
                time.sleep(0.2)
                self.keyboard_controller.type(transcribed_text)
                print("✅ Done!\n")
            else:
                print("❌ No speech detected\n")
        
        except Exception as e:
            print(f"❌ Error during transcription: {e}\n")
        
        print("Ready for next recording (Alt+S)...\n")
    
    def start_recording(self):
        """Start recording in a separate thread"""
        if not self.is_recording:
            print("\n🔴 RECORDING... (release Alt+S to stop)")
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and transcribe"""
        if self.is_recording:
            self.is_recording = False
            print("⏹️  Recording stopped")
            if self.recording_thread:
                self.recording_thread.join(timeout=1.0)
            # Transcribe in a separate thread to avoid blocking
            threading.Thread(target=self.transcribe_and_type, daemon=True).start()
    
    def on_press(self, key):
        """Handle key press events"""
        # Track Alt key
        if key in (Key.alt_l, Key.alt_r, Key.alt):
            self.alt_pressed = True
        
        # Track S key
        if hasattr(key, 'char') and key.char and key.char.lower() == 's':
            self.s_pressed = True
        
        # Start recording if both Alt and S are pressed
        if self.alt_pressed and self.s_pressed and not self.is_recording:
            self.start_recording()
    
    def on_release(self, key):
        """Handle key release events"""
        # Track Alt key release
        if key in (Key.alt_l, Key.alt_r, Key.alt):
            self.alt_pressed = False
        
        # Track S key release
        if hasattr(key, 'char') and key.char and key.char.lower() == 's':
            self.s_pressed = False
        
        # Stop recording if either Alt or S is released
        if self.is_recording and (not self.alt_pressed or not self.s_pressed):
            self.stop_recording()
    
    def start(self):
        """Start listening for keyboard events"""
        with keyboard.Listener(on_press=self.on_press, 
                              on_release=self.on_release) as listener:
            listener.join()

if __name__ == "__main__":
    try:
        typer = VoiceTyper()
        typer.start()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed required packages:")
        print("pip install openai-whisper sounddevice scipy pynput numpy")