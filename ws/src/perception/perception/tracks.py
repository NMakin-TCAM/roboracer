"""Location of the per-track run-time files (see docs/pipeline.md)."""
import os


def tracks_dir():
    # absolute: cartographer_node opens write_state's filename from its own working directory
    return os.path.abspath(os.path.expanduser(os.environ.get('TRACKS_DIR', '~/ws/tracks')))


def track_dir(name):
    return os.path.join(tracks_dir(), name)
