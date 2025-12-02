"""Analysis pipeline - orchestrates all analysis components."""
from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from ..core.models import (
    Video, Frame, Category, Medium, ClipSuggestion,
    AnalysisState, Project
)
from ..core.config import (
    VIDEO_EXTENSIONS, FRAME_SAMPLE_INTERVAL, 
    MIN_FACE_CONFIDENCE, MIN_CATEGORY_CONFIDENCE,
    CATEGORY_WEIGHTS
)
from ..core.exceptions import VideoNotFoundError, UnsupportedFormatError, AnalysisError

from .detector import MediumDetector
from .face import FaceDetector
from .categorizer import CLIPCategorizer
from .aesthetic import AestheticScorer
from .audio import AudioAnalyzer
from .technical import TechnicalScorer

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Main analysis pipeline that coordinates all components."""
    
    def __init__(self, gpu_id: int = 0):
        """Initialize the pipeline.
        
        Args:
            gpu_id: GPU to use for ML models (0 = first GPU, -1 = CPU)
        """
        self.gpu_id = gpu_id
        
        # Initialize components (lazy-loaded)
        self.detector = MediumDetector()
        self.face_detector = FaceDetector(gpu_id=gpu_id)
        self.categorizer = CLIPCategorizer()
        self.aesthetic_scorer = AestheticScorer()
        self.audio_analyzer = AudioAnalyzer()
        self.technical_scorer = TechnicalScorer()
        
        # State
        self.state = AnalysisState()
        self._progress_callback: Optional[Callable[[int, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Set callback for progress updates.
        
        Args:
            callback: Function(percent, message) to call with progress updates
        """
        self._progress_callback = callback
    
    def _report_progress(self, percent: int, message: str):
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(percent, message)
        logger.info(f"[{percent}%] {message}")
    
    def find_videos(self, folder: Path) -> list[Path]:
        """Find all video files in a folder recursively.
        
        Args:
            folder: Root folder to search
            
        Returns:
            List of video file paths
        """
        if not folder.exists():
            raise VideoNotFoundError(f"Folder not found: {folder}")
        
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(folder.rglob(f"*{ext}"))
            videos.extend(folder.rglob(f"*{ext.upper()}"))
        
        # Sort by name for consistent ordering
        videos.sort(key=lambda p: p.name.lower())
        
        logger.info(f"Found {len(videos)} video files in {folder}")
        return videos
    
    def analyze_video(self, video_path: Path) -> Video:
        """Analyze a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video object with metadata and detection results
        """
        if not video_path.exists():
            raise VideoNotFoundError(f"Video not found: {video_path}")
        
        ext = video_path.suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise UnsupportedFormatError(f"Unsupported format: {ext}")
        
        # Detect medium and LOG profile
        metadata, medium, log_profile = self.detector.analyze_video(video_path)
        
        return Video(
            path=video_path,
            metadata=metadata,
            medium=medium,
            log_profile=log_profile,
        )
    
    def extract_frame(self, video: Video, timestamp_sec: float) -> Image.Image:
        """Extract a single frame from a video.
        
        Args:
            video: Video object
            timestamp_sec: Time position in seconds
            
        Returns:
            PIL Image of the frame
        """
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp_sec),
                '-i', str(video.path),
                '-frames:v', '1',
                '-q:v', '2',
                str(tmp_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return Image.open(tmp_path).copy()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def analyze_frame(self, video: Video, timestamp_sec: float) -> Frame:
        """Analyze a single frame.
        
        Args:
            video: Video object
            timestamp_sec: Time position in seconds
            
        Returns:
            Frame object with all analysis results
        """
        # Extract frame
        image = self.extract_frame(video, timestamp_sec)
        
        # Detect faces
        faces = self.face_detector.detect(image, min_confidence=MIN_FACE_CONFIDENCE)
        
        # Categorize
        category, cat_confidence = self.categorizer.categorize(image)
        
        # Score aesthetics
        aesthetic_score = self.aesthetic_scorer.score_normalized(image)
        
        # Technical scores
        tech_scores = self.technical_scorer.score_all(image, faces)
        
        # Combine scores based on category
        weights = CATEGORY_WEIGHTS.get(category.name, {})
        
        scores = {
            'aesthetic': aesthetic_score,
            'category_confidence': cat_confidence,
            'face_count': len(faces) / 10,  # Normalize
            **tech_scores,
        }
        
        # Calculate weighted final score based on category
        final_score = 0.0
        if category == Category.PEOPLE:
            # PEOPLE: Prioritize face presence + aesthetics
            final_score = (
                scores['aesthetic'] * 0.25 +
                scores['sharpness'] * 0.20 +
                scores['composition'] * 0.20 +
                min(1.0, len(faces) * 0.3) * 0.35  # Face presence
            )
        elif category == Category.B_ROLL:
            # B_ROLL: Details, venue, nature - prioritize aesthetics + composition
            # Penalty for faces (should be no people)
            face_penalty = min(0.15, len(faces) * 0.05)
            final_score = (
                scores['aesthetic'] * 0.35 +
                scores['composition'] * 0.30 +
                scores['sharpness'] * 0.20 +
                scores['contrast'] * 0.15
            ) - face_penalty
        else:  # ARTSY
            # ARTSY: Intentional blur, light leaks, grain - prioritize aesthetics
            # Low sharpness is GOOD for artsy (inverted)
            artsy_blur_bonus = max(0, 0.15 - scores['sharpness'] * 0.15)
            final_score = (
                scores['aesthetic'] * 0.50 +
                scores['composition'] * 0.25 +
                artsy_blur_bonus +  # Bonus for intentional blur
                scores['contrast'] * 0.10
            )
        
        scores['final'] = final_score
        
        return Frame(
            video=video,
            timestamp_sec=timestamp_sec,
            category=category,
            scores=scores,
            faces=faces,
        )
    
    def analyze_video_frames(self, video: Video, 
                             sample_interval: float = FRAME_SAMPLE_INTERVAL,
                             max_frames: int = None) -> list[Frame]:
        """Analyze frames throughout a video.
        
        Args:
            video: Video object
            sample_interval: Seconds between samples
            max_frames: Maximum frames to analyze (None = no limit)
            
        Returns:
            List of Frame objects
        """
        duration = video.metadata.duration_sec
        timestamps = []
        
        t = sample_interval  # Start after first second
        while t < duration - 1:  # Stop before last second
            timestamps.append(t)
            t += sample_interval
            if max_frames and len(timestamps) >= max_frames:
                break
        
        logger.info(f"Analyzing {len(timestamps)} frames from {video.filename}")
        
        frames = []
        for i, ts in enumerate(timestamps):
            try:
                frame = self.analyze_frame(video, ts)
                frames.append(frame)
                
                # Progress update
                if i % 10 == 0:
                    pct = int((i + 1) / len(timestamps) * 100)
                    self._report_progress(pct, f"Frame {i+1}/{len(timestamps)}")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze frame at {ts}s: {e}")
                continue
        
        return frames
    
    def find_clip_suggestions(self, video: Video, frames: list[Frame]) -> list[ClipSuggestion]:
        """Find good clip candidates based on frames and audio.

        FIXED: Now generates clips per-category to ensure all categories
        get representation instead of PEOPLE dominating due to face weighting.

        Args:
            video: Video object
            frames: Analyzed frames

        Returns:
            List of ClipSuggestion objects
        """
        suggestions = []

        # Get audio events if video has audio
        audio_events = []
        if video.has_audio and video.medium != Medium.SUPER_8:
            try:
                audio_events = self.audio_analyzer.analyze(video.path)
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")

        # FIX: Generate clip suggestions PER CATEGORY
        # This prevents PEOPLE from dominating due to face presence weighting
        clips_per_category = 7  # ~20 total clips across 3 categories

        for category in [Category.PEOPLE, Category.B_ROLL, Category.ARTSY]:
            # Filter frames by category
            cat_frames = [f for f in frames if f.category == category]

            # Sort by score within category
            cat_frames_sorted = sorted(
                cat_frames,
                key=lambda f: f.scores.get('final', 0),
                reverse=True
            )

            # Take top frames for this category
            for frame in cat_frames_sorted[:clips_per_category]:
                # Use audio to find good boundaries
                if audio_events:
                    start, end = self.audio_analyzer.find_clip_boundaries(
                        audio_events, frame.timestamp_sec, duration=8.0
                    )
                else:
                    start = max(0, frame.timestamp_sec - 4)
                    end = frame.timestamp_sec + 4

                suggestions.append(ClipSuggestion(
                    video=video,
                    start_sec=start,
                    end_sec=end,
                    category=frame.category,
                    score=frame.scores.get('final', 0),
                    source='audio' if audio_events else 'visual',
                ))

        return suggestions
    
    def analyze_project(self, folder: Path) -> Project:
        """Analyze all videos in a project folder.
        
        Args:
            folder: Root folder containing videos
            
        Returns:
            Project object with all analysis results
        """
        project = Project(root_folder=folder)
        
        # Find videos
        video_paths = self.find_videos(folder)
        self.state.videos_total = len(video_paths)
        
        self._report_progress(0, f"Found {len(video_paths)} videos")
        
        for i, video_path in enumerate(video_paths):
            self.state.current_video = video_path.name
            self._report_progress(
                int(i / len(video_paths) * 100),
                f"Analyzing {video_path.name}"
            )
            
            try:
                # Analyze video
                video = self.analyze_video(video_path)
                project.videos.append(video)
                
                # Analyze frames
                frames = self.analyze_video_frames(video)
                
                # Sort into categories (PEOPLE, B_ROLL, ARTSY)
                for frame in frames:
                    if frame.category == Category.PEOPLE:
                        project.people_frames.append(frame)
                    elif frame.category == Category.B_ROLL:
                        project.broll_frames.append(frame)
                    else:  # ARTSY
                        project.artsy_frames.append(frame)

                # Find clip suggestions
                clips = self.find_clip_suggestions(video, frames)
                for clip in clips:
                    if clip.category == Category.PEOPLE:
                        project.people_clips.append(clip)
                    elif clip.category == Category.B_ROLL:
                        project.broll_clips.append(clip)
                    else:  # ARTSY
                        project.artsy_clips.append(clip)
                
                self.state.videos_processed += 1
                
                # Memory cleanup
                self._cleanup()
                
            except Exception as e:
                logger.error(f"Failed to analyze {video_path}: {e}")
                continue
        
        # Sort results by score (PEOPLE, B_ROLL, ARTSY)
        for frame_list in [project.people_frames, project.broll_frames, project.artsy_frames]:
            frame_list.sort(key=lambda f: f.scores.get('final', 0), reverse=True)

        for clip_list in [project.people_clips, project.broll_clips, project.artsy_clips]:
            clip_list.sort(key=lambda c: c.score, reverse=True)

        self._report_progress(100, "Analysis complete!")

        logger.info(
            f"Analysis complete: {len(project.people_frames)} people, "
            f"{len(project.broll_frames)} b-roll, {len(project.artsy_frames)} artsy frames"
        )
        
        return project
    
    def _cleanup(self):
        """Free memory between videos."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
