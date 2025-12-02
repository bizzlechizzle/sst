"""State persistence for save/resume functionality."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.models import AnalysisState, Project
from ..core.config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

STATE_FILENAME = 'analysis_state.json'


def get_state_path(project_folder: Path) -> Path:
    """Get path to state file for a project.
    
    Args:
        project_folder: Project root folder
        
    Returns:
        Path to state JSON file
    """
    return project_folder / '.sst' / STATE_FILENAME


def save_state(project: Project) -> Path:
    """Save project analysis state for resume.
    
    Args:
        project: Project object to save
        
    Returns:
        Path to saved state file
    """
    state_path = get_state_path(project.root_folder)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build serializable state
    state_data = {
        'version': '1.0',
        'saved_at': datetime.now().isoformat(),
        'root_folder': str(project.root_folder),
        'analysis_state': asdict(project.state),
        'videos': [str(v.path) for v in project.videos],
        'results': {
            'people_frames': len(project.people_frames),
            'details_frames': len(project.details_frames),
            'venue_frames': len(project.venue_frames),
            'people_clips': len(project.people_clips),
            'details_clips': len(project.details_clips),
            'venue_clips': len(project.venue_clips),
        },
        # Store frame references for resume
        'frame_refs': [
            {
                'video': str(f.video.path),
                'timestamp': f.timestamp_sec,
                'category': f.category.name,
                'score': f.scores.get('final', 0),
            }
            for f in project.people_frames + project.details_frames + project.venue_frames
        ],
    }
    
    with open(state_path, 'w') as f:
        json.dump(state_data, f, indent=2)
    
    logger.info(f"Saved state to {state_path}")
    return state_path


def load_state(project_folder: Path) -> Optional[dict]:
    """Load saved analysis state.
    
    Args:
        project_folder: Project root folder
        
    Returns:
        State dictionary or None if no state found
    """
    state_path = get_state_path(project_folder)
    
    if not state_path.exists():
        logger.debug(f"No saved state found at {state_path}")
        return None
    
    try:
        with open(state_path, 'r') as f:
            state_data = json.load(f)
        
        logger.info(f"Loaded state from {state_path}")
        logger.info(f"  Saved at: {state_data.get('saved_at')}")
        logger.info(f"  Videos: {len(state_data.get('videos', []))}")
        logger.info(f"  Frames: {len(state_data.get('frame_refs', []))}")
        
        return state_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid state file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return None


def clear_state(project_folder: Path) -> bool:
    """Remove saved state file.
    
    Args:
        project_folder: Project root folder
        
    Returns:
        True if state was cleared
    """
    state_path = get_state_path(project_folder)
    
    if state_path.exists():
        state_path.unlink()
        logger.info(f"Cleared state at {state_path}")
        return True
    
    return False


def has_saved_state(project_folder: Path) -> bool:
    """Check if a project has saved state.
    
    Args:
        project_folder: Project root folder
        
    Returns:
        True if saved state exists
    """
    return get_state_path(project_folder).exists()


def get_state_info(project_folder: Path) -> Optional[dict]:
    """Get summary info about saved state without full load.
    
    Args:
        project_folder: Project root folder
        
    Returns:
        Dictionary with state summary or None
    """
    state = load_state(project_folder)
    if not state:
        return None
    
    return {
        'saved_at': state.get('saved_at'),
        'videos_count': len(state.get('videos', [])),
        'frames_count': len(state.get('frame_refs', [])),
        'progress': state.get('analysis_state', {}).get('videos_processed', 0),
        'total': state.get('analysis_state', {}).get('videos_total', 0),
    }
