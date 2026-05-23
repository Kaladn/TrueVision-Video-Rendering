"""TrueAudio sibling state runtime.

TrueAudio listens as audio-state. It is intentionally separate from
TrueVision, which sees visual state.
"""

from .logging import log_machine_pre_sound_state, log_pre_sound_state
from .lyrics import align_lyrics_to_speech_segments
from .replay import replay_trueaudio_state
from .replayable import (
    log_file_replayable_audio_state,
    log_machine_replayable_audio_state,
    replay_replayable_audio_state,
    write_replayable_audio_state,
)
from .speech import detect_speech_segments_from_replayable_state

__all__ = [
    "align_lyrics_to_speech_segments",
    "log_file_replayable_audio_state",
    "log_machine_pre_sound_state",
    "log_machine_replayable_audio_state",
    "log_pre_sound_state",
    "replay_replayable_audio_state",
    "replay_trueaudio_state",
    "detect_speech_segments_from_replayable_state",
    "write_replayable_audio_state",
]
