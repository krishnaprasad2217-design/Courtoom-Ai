"""
Audio recording module for lie detection system.
Records microphone input during analysis and saves to WAV file.
Uses InputStream so recording can be stopped early at any time.
"""

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import threading
import time
from pathlib import Path


class AudioRecorder:
    def __init__(self, sample_rate=16000, duration=40):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Recording sample rate (Hz)
            duration: Max recording duration (seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.recording = None
        self.is_recording = False
        self.output_file = "temp_audio.wav"
        self._chunks = []
        self._stream = None
        self._stop_event = threading.Event()

    def start_recording(self):
        """Start recording audio in background thread"""
        self._chunks = []
        self._stop_event.clear()
        self.is_recording = True
        self.recording = None
        threading.Thread(target=self._record_audio, daemon=True).start()
        print(f"🎤 Audio recording started (up to {self.duration}s)...")

    def stop_recording(self):
        """Stop recording immediately — saves whatever was captured so far"""
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        print("🛑 Audio recording stopped early")

    def _record_audio(self):
        """Internal: record using InputStream so we can stop at any time"""
        try:
            def _callback(indata, frames, time_info, status):
                if not self._stop_event.is_set():
                    self._chunks.append(indata.copy())

            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                dtype=np.float32, callback=_callback) as stream:
                self._stream = stream
                # Wait until stop event or duration reached
                self._stop_event.wait(timeout=self.duration)

            # Combine all chunks
            if self._chunks:
                self.recording = np.concatenate(self._chunks, axis=0)
                print(f"✅ Recording complete ({len(self.recording)} samples, "
                      f"{len(self.recording)/self.sample_rate:.1f}s)")
            else:
                self.recording = np.array([])
                print("⚠️  No audio chunks captured")

        except Exception as e:
            print(f"❌ Recording error: {e}")
            self.recording = np.array([])
        finally:
            self.is_recording = False
            self._stream = None

    def save_recording(self):
        """Save recorded audio to WAV file — waits briefly if still recording"""
        # Wait up to 3s for thread to wrap up
        wait_time = 0
        while self.is_recording and wait_time < 3.0:
            time.sleep(0.1)
            wait_time += 0.1

        if self.recording is None or (hasattr(self.recording, '__len__') and len(self.recording) == 0):
            print("❌ No audio recorded")
            return None

        try:
            audio_data = self.recording
            # Flatten 2D (samples x channels) → 1D
            if hasattr(audio_data, 'ndim') and audio_data.ndim == 2:
                audio_data = audio_data[:, 0]

            audio_int = np.int16(np.clip(audio_data, -1.0, 1.0) * 32767)
            wavfile.write(self.output_file, self.sample_rate, audio_int)
            print(f"✅ Audio saved to {self.output_file} ({len(audio_int)/self.sample_rate:.1f}s)")
            return self.output_file

        except Exception as e:
            print(f"❌ Save error: {e}")
            return None

    def cleanup(self):
        """Delete temporary audio file"""
        try:
            Path(self.output_file).unlink(missing_ok=True)
            print(f"🗑️  Cleaned up {self.output_file}")
        except Exception as e:
            print(f"Cleanup warning: {e}")

