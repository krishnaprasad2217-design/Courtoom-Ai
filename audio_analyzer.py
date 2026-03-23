"""
Audio analysis module using OpenAI Whisper for speech-to-text.
Transcribes audio and analyzes speech patterns for lie detection.
"""

import whisper
import os
from pathlib import Path


class AudioAnalyzer:
    def __init__(self, model_size="base"):
        """
        Initialize audio analyzer with Whisper model.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self.transcript = ""
        self.word_count = 0
        self.speech_rate = 0.0  # words per second
        self.audio_flag = 0
        self.word_threshold = 7  # Threshold for word count
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model (lazy loading on first use)"""
        try:
            if self.model is None:
                print(f"📥 Loading Whisper ({self.model_size}) model...")
                self.model = whisper.load_model(self.model_size)
                print(f"✅ Whisper model loaded")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            raise
    
    def transcribe_audio(self, audio_file, duration=40):
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_file: Path to WAV audio file
            duration: Duration of audio in seconds (for speech rate calc). If None, will be auto-detected.
            
        Returns:
            dict with transcript, word_count, speech_rate, audio_flag
        """
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return None
        
        try:
            print(f"🎵 Transcribing audio ({os.path.basename(audio_file)})...")
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio_file, language="en")
            self.transcript = result["text"].strip()
            
            print(f"📝 Transcript: {self.transcript}")
            
            # Count words
            self.word_count = len(self.transcript.split()) if self.transcript else 0
            
            # Auto-detect duration from Whisper result if not provided
            if duration is None or duration <= 0:
                # Whisper returns duration in the result
                duration = result.get("duration", 40)
                print(f"📊 Audio duration: {duration:.1f}s")
            
            # Calculate speech rate (words per second) - safely handle division
            if duration and duration > 0:
                self.speech_rate = self.word_count / duration
            else:
                self.speech_rate = 0.0
            
            # Determine audio flag based on word count
            # If words > threshold → NORMAL (audio_flag=0)
            # If words ≤ threshold → LIE/SUSPICIOUS (audio_flag=1)
            self.audio_flag = 0 if self.word_count > self.word_threshold else 1
            
            print(f"📊 Word Count: {self.word_count} | Speech Rate: {self.speech_rate:.2f} wps")
            print(f"🚩 Audio Flag: {self.audio_flag} ({'NORMAL' if self.audio_flag == 0 else 'SUSPICIOUS'})")
            
            return {
                'transcript': self.transcript,
                'word_count': self.word_count,
                'speech_rate': self.speech_rate,
                'audio_flag': self.audio_flag,
                'threshold': self.word_threshold
            }
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_threshold(self, threshold):
        """Adjust word count threshold"""
        self.word_threshold = threshold
        print(f"⚙️  Audio threshold set to {threshold} words")
    
    def reset(self):
        """Reset analyzer state"""
        self.transcript = ""
        self.word_count = 0
        self.speech_rate = 0.0
        self.audio_flag = 0
