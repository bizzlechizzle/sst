"""Data models for SST."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional


class Medium(Enum):
    """The type of camera used to shoot the video."""
    SUPER_8 = auto()   # Film scans - high res, film frame rates, NO audio
    DAD_CAM = auto()   # Consumer cameras - SD to HD, HAS audio
    MODERN = auto()    # Professional cameras - 4K+, HAS audio


class Category(Enum):
    """What the frame/clip is about - 3 main categories."""
    PEOPLE = auto()    # Humans are the main subject (faces detected)
    B_ROLL = auto()    # Non-people content: venue, details, dress, flowers, nature
    ARTSY = auto()     # Stylized footage: blur, light leaks, grain, bokeh


class SubCategory(Enum):
    """Fine-grained classification within each category."""
    # PEOPLE sub-categories
    BRIDE = "bride"              # Solo bride shots
    GROOM = "groom"              # Solo groom shots
    COUPLE = "couple"            # Bride + groom together
    FAMILY = "family"            # Parents, siblings, grandparents
    FRIENDS = "friends"          # Wedding guests, general attendees
    PARTY = "party"              # Bridesmaids, groomsmen, wedding party
    KIDS = "kids"                # Flower girl, ring bearer, children
    SOLO = "solo"                # Single person (not bride/groom)

    # B_ROLL sub-categories
    DETAILS = "details"          # Rings, invitations, cake, table settings
    DRESS = "dress"              # Dress on hanger, shoes, accessories, cufflinks
    FLOWERS = "flowers"          # Bouquet, arrangements, boutonniere
    VENUE_INTERIOR = "venue_interior"  # Ceremony room, reception hall
    VENUE_EXTERIOR = "venue_exterior"  # Building exterior, entrance, gardens
    NATURE = "nature"            # Trees, grass, sunset, water
    FOOD = "food"                # Champagne, appetizers, dinner
    DECOR = "decor"              # Signage, table numbers, string lights

    # ARTSY sub-categories
    MOTION_BLUR = "motion_blur"  # Shutter drag, spinning dancers
    LIGHT_LEAK = "light_leak"    # Film light leak effect
    BOKEH = "bokeh"              # Shallow DOF, blurry background lights
    GRAIN = "grain"              # Film grain, vintage texture
    FLARE = "flare"              # Lens flare, sun flare
    SILHOUETTE = "silhouette"    # Backlit silhouette
    REFLECTION = "reflection"    # Mirror, water, glass reflection
    ABSTRACT = "abstract"        # Creative angles, unusual perspective


class LogProfile(Enum):
    """LOG color profile for footage that needs a LUT."""
    NONE = auto()      # Standard/Rec.709 - no LUT needed
    SLOG3 = auto()     # Sony S-Log3
    CLOG3 = auto()     # Canon C-Log3
    VLOG = auto()      # Panasonic V-Log
    NLOG = auto()      # Nikon N-Log
    FLAT = auto()      # Generic flat profile


class AudioEventType(Enum):
    """Types of audio events we can detect."""
    SPEECH = "speech"
    LAUGHTER = "laughter"
    APPLAUSE = "applause"
    CHEER = "cheer"
    MUSIC = "music"
    SILENCE = "silence"


@dataclass
class BoundingBox:
    """A rectangle, used for face locations and crops.
    
    Coordinates are in pixels from top-left corner.
    """
    x: int       # Left edge
    y: int       # Top edge
    width: int
    height: int
    
    @property
    def center_x(self) -> int:
        """X coordinate of the center."""
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        """Y coordinate of the center."""
        return self.y + self.height // 2
    
    @property
    def area(self) -> int:
        """Area in pixels."""
        return self.width * self.height
    
    @property
    def right(self) -> int:
        """Right edge X coordinate."""
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        """Bottom edge Y coordinate."""
        return self.y + self.height
    
    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def as_xyxy(self) -> tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.right, self.bottom)


@dataclass
class Face:
    """A detected face in a frame."""
    bbox: BoundingBox
    confidence: float          # 0.0 to 1.0
    landmarks: Optional[dict] = None  # Eye positions, etc.
    
    @property
    def size(self) -> int:
        """Face size in pixels (area)."""
        return self.bbox.area


@dataclass
class VideoMetadata:
    """Technical information about a video file."""
    width: int
    height: int
    fps: float
    duration_sec: float
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    color_space: Optional[str] = None
    gamma: Optional[str] = None  # For detecting LOG profiles
    
    @property
    def resolution(self) -> str:
        """Human-readable resolution string."""
        if self.width >= 3840:
            return "4K"
        elif self.width >= 1920:
            return "1080p"
        elif self.width >= 1280:
            return "720p"
        elif self.width >= 720:
            return "SD"
        else:
            return f"{self.width}x{self.height}"
    
    @property
    def frame_count(self) -> int:
        """Total number of frames in video."""
        return int(self.duration_sec * self.fps)


@dataclass
class Video:
    """A video file and its analysis results."""
    path: Path
    metadata: VideoMetadata
    medium: Medium = Medium.MODERN
    log_profile: LogProfile = LogProfile.NONE
    
    @property
    def has_audio(self) -> bool:
        """Whether this video has an audio track."""
        return self.metadata.has_audio
    
    @property
    def needs_lut(self) -> bool:
        """Whether this video needs a LUT applied."""
        return self.log_profile != LogProfile.NONE
    
    @property
    def filename(self) -> str:
        """Just the filename without path."""
        return self.path.name
    
    @property
    def stem(self) -> str:
        """Filename without extension."""
        return self.path.stem


@dataclass
class Frame:
    """A single frame extracted from a video."""
    video: Video
    timestamp_sec: float       # Position in video
    category: Category
    sub_category: Optional[SubCategory] = None  # Fine-grained classification
    scores: dict = field(default_factory=dict)  # Various quality scores
    faces: list[Face] = field(default_factory=list)
    
    @property
    def frame_number(self) -> int:
        """Frame number in the video."""
        return int(self.timestamp_sec * self.video.metadata.fps)
    
    @property
    def total_score(self) -> float:
        """Combined score from all factors."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    @property
    def has_faces(self) -> bool:
        """Whether any faces were detected."""
        return len(self.faces) > 0
    
    @property
    def face_count(self) -> int:
        """Number of faces detected."""
        return len(self.faces)
    
    @property
    def timecode(self) -> str:
        """Human-readable timecode (HH:MM:SS.mmm)."""
        hours = int(self.timestamp_sec // 3600)
        minutes = int((self.timestamp_sec % 3600) // 60)
        seconds = self.timestamp_sec % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


@dataclass
class AudioEvent:
    """An interesting moment detected in audio."""
    type: AudioEventType
    start_sec: float
    end_sec: float
    confidence: float
    energy: float = 0.0  # Loudness level
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_sec - self.start_sec
    
    @property
    def midpoint(self) -> float:
        """Midpoint timestamp."""
        return (self.start_sec + self.end_sec) / 2


@dataclass
class ClipSuggestion:
    """A suggested video clip to export."""
    video: Video
    start_sec: float
    end_sec: float
    category: Category
    score: float
    source: str  # "audio" or "visual" or "scene"
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_sec - self.start_sec
    
    @property
    def timecode_start(self) -> str:
        """Human-readable start timecode."""
        minutes = int(self.start_sec // 60)
        seconds = self.start_sec % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    
    @property
    def timecode_end(self) -> str:
        """Human-readable end timecode."""
        minutes = int(self.end_sec // 60)
        seconds = self.end_sec % 60
        return f"{minutes:02d}:{seconds:05.2f}"


@dataclass
class ExportSettings:
    """Settings for exporting screenshots and clips."""
    output_folder: Path
    crops: list[str] = field(default_factory=lambda: ['square', 'portrait', 'story'])
    jpeg_quality: int = 90
    video_crf: int = 22
    lut_path: Optional[Path] = None
    quotas: dict = field(default_factory=dict)
    clip_duration_short: float = 8.0
    clip_duration_long: float = 30.0


@dataclass
class AnalysisState:
    """State of video analysis, used for save/resume."""
    videos_total: int = 0
    videos_processed: int = 0
    current_video: Optional[str] = None
    current_frame: int = 0
    frames_found: int = 0
    clips_found: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Overall progress as percentage."""
        if self.videos_total == 0:
            return 0.0
        return (self.videos_processed / self.videos_total) * 100


@dataclass
class Project:
    """A complete project with all data."""
    root_folder: Path
    videos: list[Video] = field(default_factory=list)
    settings: ExportSettings = None

    # Categorized results (PEOPLE, B_ROLL, ARTSY)
    people_frames: list[Frame] = field(default_factory=list)
    people_clips: list[ClipSuggestion] = field(default_factory=list)
    broll_frames: list[Frame] = field(default_factory=list)
    broll_clips: list[ClipSuggestion] = field(default_factory=list)
    artsy_frames: list[Frame] = field(default_factory=list)
    artsy_clips: list[ClipSuggestion] = field(default_factory=list)

    # Analysis state
    state: AnalysisState = field(default_factory=AnalysisState)

    def get_frames_by_category(self, category: Category) -> list[Frame]:
        """Get all frames for a category."""
        if category == Category.PEOPLE:
            return self.people_frames
        elif category == Category.B_ROLL:
            return self.broll_frames
        else:  # ARTSY
            return self.artsy_frames

    def get_clips_by_category(self, category: Category) -> list[ClipSuggestion]:
        """Get all clips for a category."""
        if category == Category.PEOPLE:
            return self.people_clips
        elif category == Category.B_ROLL:
            return self.broll_clips
        else:  # ARTSY
            return self.artsy_clips
