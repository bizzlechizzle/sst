"""Audio analysis for moment detection."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..core.models import AudioEvent, AudioEventType
from ..core.exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyzes audio to find interesting moments."""
    
    def __init__(self):
        self._vad_model = None
        self._vad_utils = None
        self._sample_rate = 16000  # Silero VAD needs 16kHz
        self._initialized = False
    
    def _load_vad(self):
        """Load Silero VAD (Voice Activity Detection) model."""
        if self._initialized:
            return
        
        logger.info("Loading Silero VAD model...")
        
        try:
            model, utils = torch.hub.load(
                'snakers4/silero-vad',
                'silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_utils = utils
            self._initialized = True
            
            logger.info("Silero VAD loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}")
            # Continue without VAD - will skip speech detection
    
    def extract_audio(self, video_path: Path, output_path: Optional[Path] = None) -> np.ndarray:
        """Extract audio from video as numpy array.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save WAV file
        
        Returns:
            Audio samples as numpy array (mono, 16kHz)
            
        Raises:
            AudioExtractionError: If extraction fails
        """
        import soundfile as sf
        
        # Create temp file if no output specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
            cleanup = True
        else:
            cleanup = False
        
        # Extract audio with FFmpeg
        cmd = [
            'ffmpeg', '-y',           # Overwrite output
            '-i', str(video_path),    # Input video
            '-vn',                     # No video
            '-acodec', 'pcm_s16le',   # WAV format
            '-ar', str(self._sample_rate),  # 16kHz
            '-ac', '1',               # Mono
            str(output_path)
        ]
        
        logger.debug(f"Extracting audio from {video_path}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(f"FFmpeg audio extraction failed: {e.stderr.decode()}")
        
        # Load the audio
        try:
            audio, sr = sf.read(output_path)
        except Exception as e:
            raise AudioExtractionError(f"Failed to read extracted audio: {e}")
        finally:
            # Clean up temp file
            if cleanup and output_path.exists():
                output_path.unlink()
        
        return audio.astype(np.float32)
    
    def detect_speech(self, audio: np.ndarray) -> list[AudioEvent]:
        """Detect speech segments using Silero VAD.
        
        Args:
            audio: Audio samples (mono, 16kHz)
        
        Returns:
            List of AudioEvent objects for speech segments
        """
        self._load_vad()
        
        if self._vad_model is None:
            logger.warning("VAD not available, skipping speech detection")
            return []
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio)
        
        try:
            # Get speech timestamps using utils
            get_speech_timestamps = self._vad_utils[0]
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=self._sample_rate,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )
        except Exception as e:
            logger.warning(f"Speech detection failed: {e}")
            return []
        
        events = []
        for segment in speech_timestamps:
            start_sec = segment['start'] / self._sample_rate
            end_sec = segment['end'] / self._sample_rate
            
            events.append(AudioEvent(
                type=AudioEventType.SPEECH,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=0.9,  # VAD is quite reliable
            ))
        
        logger.debug(f"Detected {len(events)} speech segments")
        return events
    
    def detect_energy_peaks(self, audio: np.ndarray, threshold: float = 0.7) -> list[AudioEvent]:
        """Detect high-energy moments (laughter, applause, cheers).
        
        Uses RMS (Root Mean Square) energy to find loud moments.
        
        Args:
            audio: Audio samples (mono)
            threshold: Energy threshold (0-1, relative to max)
            
        Returns:
            List of AudioEvent objects
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed, skipping energy detection")
            return []
        
        # Compute RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Convert to dB for easier thresholding
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find peaks above threshold
        threshold_db = np.max(rms_db) * threshold
        peak_frames = np.where(rms_db > threshold_db)[0]
        
        if len(peak_frames) == 0:
            return []
        
        # Group consecutive frames into segments
        events = []
        segment_start = peak_frames[0]
        prev_frame = peak_frames[0]
        
        for frame in peak_frames[1:]:
            # If there's a gap, end the segment
            if frame - prev_frame > 10:  # ~0.3 seconds gap
                start_sec = segment_start * hop_length / self._sample_rate
                end_sec = prev_frame * hop_length / self._sample_rate
                
                # Classify the event based on spectral features
                event_type = self._classify_energy_event(audio, start_sec, end_sec)
                
                events.append(AudioEvent(
                    type=event_type,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    confidence=0.7,
                    energy=float(np.mean(rms_db[segment_start:prev_frame+1])),
                ))
                
                segment_start = frame
            
            prev_frame = frame
        
        # Don't forget the last segment
        if prev_frame > segment_start:
            start_sec = segment_start * hop_length / self._sample_rate
            end_sec = prev_frame * hop_length / self._sample_rate
            event_type = self._classify_energy_event(audio, start_sec, end_sec)
            
            events.append(AudioEvent(
                type=event_type,
                start_sec=start_sec,
                end_sec=end_sec,
                confidence=0.7,
                energy=float(np.mean(rms_db[segment_start:prev_frame+1])),
            ))
        
        logger.debug(f"Detected {len(events)} energy peaks")
        return events
    
    def _classify_energy_event(self, audio: np.ndarray, start_sec: float, end_sec: float) -> AudioEventType:
        """Classify what type of energy event this is.
        
        Uses spectral features to distinguish:
        - Applause: Broadband noise (flat spectrum)
        - Laughter: Mid-frequency variations
        - Cheering: High energy, harmonic content
        """
        try:
            import librosa
        except ImportError:
            return AudioEventType.CHEER
        
        # Extract the segment
        start_sample = int(start_sec * self._sample_rate)
        end_sample = int(end_sec * self._sample_rate)
        segment = audio[start_sample:end_sample]
        
        if len(segment) < 1024:
            return AudioEventType.CHEER  # Default for short segments
        
        # Compute spectral features
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=segment))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self._sample_rate))
        
        # Applause: Very flat spectrum (noise-like)
        if spectral_flatness > 0.3:
            return AudioEventType.APPLAUSE
        
        # Laughter: Mid-range centroid with variation
        if 1000 < spectral_centroid < 3000:
            return AudioEventType.LAUGHTER
        
        # Default to cheer
        return AudioEventType.CHEER
    
    def detect_music(self, audio: np.ndarray) -> list[AudioEvent]:
        """Detect music segments.
        
        Uses tempo and harmonic content to identify music.
        
        Args:
            audio: Audio samples (mono)
            
        Returns:
            List of AudioEvent objects for music segments
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed, skipping music detection")
            return []
        
        # Compute chromagram (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self._sample_rate)
        
        # Music has consistent harmonic patterns
        chroma_std = np.std(chroma, axis=0)
        
        # Low variation = likely music
        music_frames = chroma_std < np.percentile(chroma_std, 30)
        
        # Convert to time segments
        hop_length = 512
        events = []
        
        # Group consecutive frames
        in_music = False
        start_frame = 0
        
        for i, is_music in enumerate(music_frames):
            if is_music and not in_music:
                start_frame = i
                in_music = True
            elif not is_music and in_music:
                # End of music segment
                if i - start_frame > 20:  # At least 0.5 seconds
                    events.append(AudioEvent(
                        type=AudioEventType.MUSIC,
                        start_sec=start_frame * hop_length / self._sample_rate,
                        end_sec=i * hop_length / self._sample_rate,
                        confidence=0.6,
                    ))
                in_music = False
        
        logger.debug(f"Detected {len(events)} music segments")
        return events
    
    def analyze(self, video_path: Path) -> list[AudioEvent]:
        """Full audio analysis of a video.
        
        Returns all detected audio events sorted by time.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of AudioEvent objects sorted by start time
        """
        logger.info(f"Analyzing audio: {video_path.name}")
        
        # Extract audio
        try:
            audio = self.extract_audio(video_path)
        except AudioExtractionError as e:
            logger.warning(f"Could not extract audio: {e}")
            return []
        
        # Detect all event types
        events = []
        events.extend(self.detect_speech(audio))
        events.extend(self.detect_energy_peaks(audio))
        events.extend(self.detect_music(audio))
        
        # Sort by start time
        events.sort(key=lambda e: e.start_sec)
        
        logger.info(f"Found {len(events)} audio events")
        return events
    
    def find_clip_boundaries(self, events: list[AudioEvent], target_time: float, 
                            duration: float = 8.0) -> tuple[float, float]:
        """Find good clip boundaries near a target time.
        
        Tries to end clips at natural breaks (end of speech, after laughter, etc.)
        
        Args:
            events: List of audio events
            target_time: Approximate time to center clip around
            duration: Desired clip duration
            
        Returns:
            Tuple of (start_sec, end_sec)
        """
        half_duration = duration / 2
        
        # Default boundaries
        start_sec = max(0, target_time - half_duration)
        end_sec = target_time + half_duration
        
        # Look for good end points
        for event in events:
            # Prefer ending after speech ends
            if event.type == AudioEventType.SPEECH:
                if abs(event.end_sec - end_sec) < 2.0:  # Within 2 seconds
                    end_sec = event.end_sec + 0.5  # Small buffer
                    start_sec = end_sec - duration
                    break
            
            # Or after laughter/applause
            if event.type in (AudioEventType.LAUGHTER, AudioEventType.APPLAUSE):
                if abs(event.end_sec - end_sec) < 2.0:
                    end_sec = event.end_sec + 0.3
                    start_sec = end_sec - duration
                    break
        
        return max(0, start_sec), end_sec
