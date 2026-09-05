#!/usr/bin/env python3
"""Gesture music in one file: wave your hand, hear a melody.

Everything needed is in this single script — camera, hand tracking, note
choice, synthesis and audio out. Save it anywhere and run it:

    python one_file_gesture_music.py

It needs only what the main project already installs:

    pip install mediapipe opencv-python numpy sounddevice

How to play it:

    right hand, left <-> right   the note (left is low)
    right hand, up               louder
    right hand out of frame      silence
    left hand, up                brighter (opens the filter)
    q or Esc                     quit

This is the short version of the full project, kept to one file so there is
nothing to configure. The full version adds MIDI output to a DAW, JSON mapping
files, two-handed gestures, hot reload and an on-screen meter panel; see
main.py. Everything here is deliberately simple rather than clever.
"""

from __future__ import annotations

import math
import queue
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Settings — the whole instrument is these ten numbers.
# --------------------------------------------------------------------------
CAMERA = 0                  # webcam index, or an "rtsp://..." URL
ROOT_NOTE = 57              # A3: the lowest note your hand can reach
SCALE = (0, 3, 5, 7, 10)    # pentatonic minor: no wrong notes
OCTAVES = 2
SAMPLE_RATE = 44100
BLOCK = 256                 # ~6 ms of audio latency
SMOOTHING = 0.35            # 0..1, lower is smoother but laggier
HYSTERESIS = 0.25           # how far past a note boundary before it changes
GAIN = 0.3
FRAME_SIZE = (1280, 720)    # smaller is faster; 1920x1080 halves the frame rate


def scale_notes() -> list:
    """The ladder of MIDI notes the hand can reach."""
    notes = []
    for step in range(int(len(SCALE) * OCTAVES) + 1):
        octave, degree = divmod(step, len(SCALE))
        note = ROOT_NOTE + octave * 12 + SCALE[degree]
        if note <= 127:
            notes.append(note)
    return notes


NOTES = scale_notes()


# --------------------------------------------------------------------------
# Synth: two detuned saws through a resonant low-pass filter.
# --------------------------------------------------------------------------
class Synth:
    """Small subtractive synth. ``render`` is pure, so it can be tested."""

    def __init__(self, samplerate=SAMPLE_RATE, gain=GAIN):
        self.samplerate = float(samplerate)
        self.gain = gain
        self.commands: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
        self.stream = None

        self.frequency = 0.0
        self.target_level = 0.0
        self.level = 0.0
        self.cutoff = 900.0
        self.target_cutoff = 900.0
        self.phases = np.array([0.0, 0.4])
        self.low = self.band = 0.0

    # -- control (called from the video thread) ---------------------------
    def note_on(self, note: int) -> None:
        self.commands.put(("note", float(440.0 * 2 ** ((note - 69) / 12))))

    def note_off(self) -> None:
        self.commands.put(("off", 0.0))

    def set_cutoff(self, hz: float) -> None:
        self.commands.put(("cutoff", float(hz)))

    def _drain(self) -> None:
        while True:
            try:
                name, value = self.commands.get_nowait()
            except queue.Empty:
                return
            if name == "note":
                self.frequency = value
                self.target_level = 1.0
            elif name == "off":
                self.target_level = 0.0
            elif name == "cutoff":
                self.target_cutoff = value

    # -- audio ------------------------------------------------------------
    def render(self, frames: int) -> np.ndarray:
        """One block of stereo float32 audio."""
        self._drain()
        if frames <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Envelope: fast attack so notes feel immediate, slower release so
        # letting go is not a click.
        seconds = 0.01 if self.target_level > self.level else 0.15
        alpha = 1.0 - math.exp(-frames / (seconds * self.samplerate))
        end_level = self.level + (self.target_level - self.level) * alpha
        envelope = np.linspace(self.level, end_level, frames)
        self.level = end_level

        mix = np.zeros(frames)
        if self.frequency > 0.0 and (self.level > 1e-4 or self.target_level > 0.0):
            for index, detune in enumerate((0.997, 1.004)):   # slight detune = width
                increment = self.frequency * detune / self.samplerate
                if not 0.0 < increment < 0.5:
                    continue
                phase = (self.phases[index] + increment * np.arange(1, frames + 1)) % 1.0
                mix += 2.0 * phase - 1.0            # sawtooth
                self.phases[index] = phase[-1]
            mix *= 0.5 * envelope

        # Chamberlin state-variable low-pass, glided towards the target cutoff.
        cutoff = np.linspace(self.cutoff, self.target_cutoff, frames)
        self.cutoff = self.target_cutoff
        out = np.empty(frames)
        low, band, damping = self.low, self.band, 1.0 / 1.6
        for i in range(frames):
            f = 2.0 * math.sin(math.pi * min(cutoff[i], self.samplerate / 3) / self.samplerate)
            high = mix[i] - low - damping * band
            band += f * high
            low += f * band
            out[i] = low
        self.low, self.band = (low, band) if math.isfinite(low + band) else (0.0, 0.0)

        out = np.tanh(out * self.gain * 2.0)
        return np.column_stack([out, out]).astype(np.float32)

    # -- device -----------------------------------------------------------
    def start(self) -> "Synth":
        import sounddevice as sd

        def callback(outdata, frames, _time, _status):
            outdata[:] = self.render(frames)

        self.stream = sd.OutputStream(
            samplerate=self.samplerate, blocksize=BLOCK, channels=2,
            dtype="float32", callback=callback,
        )
        self.stream.start()
        return self

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# --------------------------------------------------------------------------
# Hand tracking — works with both MediaPipe generations.
# --------------------------------------------------------------------------
def open_tracker():
    """Return ``(process, close)``. ``process(rgb)`` -> list of (21, 3) arrays."""
    import mediapipe as mp

    if hasattr(mp, "solutions"):        # MediaPipe 0.10.x, the macOS-safe one
        hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

        def process(rgb):
            result = hands.process(rgb)
            found = []
            for marks, handed in zip(result.multi_hand_landmarks or [],
                                     result.multi_handedness or []):
                label = handed.classification[0].label
                # 0.10.x assumes a mirrored image, and we mirror ours, so the
                # label is already the player's real hand.
                points = np.array([[p.x, p.y, p.z] for p in marks.landmark])
                found.append((label.lower(), points))
            return found

        return process, hands.close

    # MediaPipe 1.x: Tasks API, needs the model file downloaded once.
    import urllib.request
    from pathlib import Path
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = Path.home() / ".cache" / "gesture-midi-controller" / "hand_landmarker.task"
    if not model.is_file():
        model.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the hand model once (~7.5 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task", model)

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
        )
    )
    clock = {"ms": 0}

    def process(rgb):
        clock["ms"] += 33
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, clock["ms"])
        found = []
        for marks, handed in zip(result.hand_landmarks, result.handedness):
            # 1.x labels the un-mirrored hand, so flip it: we mirror our frames.
            label = "right" if handed[0].category_name.lower().startswith("l") else "left"
            found.append((label, np.array([[p.x, p.y, p.z] for p in marks])))
        return found

    return process, landmarker.close


# --------------------------------------------------------------------------
def main() -> int:
    camera = cv2.VideoCapture(CAMERA)
    if not camera.isOpened():
        print(f"could not open camera {CAMERA!r} — try 1, or 2")
        return 2
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    process, close_tracker = open_tracker()
    try:
        synth = Synth().start()
    except Exception as exc:
        print(f"no audio output: {exc}")
        return 2

    position, brightness, index, playing = 0.5, 0.5, None, False
    print("playing — move your right hand left and right. q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)               # mirror, so it feels natural
            hands = dict(process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            right, left = hands.get("right"), hands.get("left")

            if right is not None:
                # Smooth the raw wrist position: landmarks jitter every frame.
                position += (float(right[0, 0]) - position) * SMOOTHING
                spot = min(max((position - 0.12) / 0.76, 0.0), 1.0) * (len(NOTES) - 1)
                # Only change note once the hand is properly past the boundary,
                # otherwise a resting hand stutters between two pitches.
                if index is None or abs(spot - index) > 0.5 + HYSTERESIS:
                    index = int(round(spot))
                    synth.note_on(NOTES[index])
                    playing = True
            elif playing:
                synth.note_off()
                playing, index = False, None

            if left is not None:
                brightness += ((1.0 - float(left[0, 1])) - brightness) * SMOOTHING
            synth.set_cutoff(180.0 * (40.0 ** min(max(brightness, 0.0), 1.0)))

            for _, points in hands.items():
                height, width = frame.shape[:2]
                for x, y, _z in points:
                    cv2.circle(frame, (int(x * width), int(y * height)), 4, (120, 220, 255), -1)
            note_name = NOTES[index] if index is not None and playing else "-"
            cv2.putText(frame, f"note {note_name}   cutoff {int(180.0 * 40.0 ** brightness)} Hz",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Gesture music", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        synth.note_off()
        synth.render(BLOCK)
        synth.stop()
        close_tracker()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
