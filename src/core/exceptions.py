"""Custom exceptions for SST."""


class SSTError(Exception):
    """Base exception for all SST errors."""
    pass


class VideoNotFoundError(SSTError):
    """Raised when a video file doesn't exist."""
    pass


class UnsupportedFormatError(SSTError):
    """Raised when a video format isn't supported."""
    pass


class FFmpegError(SSTError):
    """Raised when FFmpeg fails."""
    pass


class ModelLoadError(SSTError):
    """Raised when an ML model fails to load."""
    pass


class NoFacesFoundError(SSTError):
    """Raised when no faces are found (and faces were expected)."""
    pass


class AudioExtractionError(SSTError):
    """Raised when audio can't be extracted from video."""
    pass


class AnalysisError(SSTError):
    """Raised when video analysis fails."""
    pass


class ExportError(SSTError):
    """Raised when export fails."""
    pass


class ConfigurationError(SSTError):
    """Raised when configuration is invalid."""
    pass
