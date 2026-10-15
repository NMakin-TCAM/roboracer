"""Location of the per-track run-time files (see docs/pipeline.md)."""
import os


def tracks_dir():
    return os.environ.get('TRACKS_DIR', os.path.expanduser('~/ws/tracks'))


def track_dir(name):
    return os.path.join(tracks_dir(), name)
