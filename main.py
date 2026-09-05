#!/usr/bin/env python3
"""Musical Instrument Gesture Controller — camera hand tracking -> MIDI CC.

Point a camera at your hands, and every gesture becomes a MIDI control change
that Logic, Ableton, GarageBand, rekordbox or a hardware synth can follow.

Examples
--------
    # USB webcam, virtual MIDI port, preview window
    python main.py --camera 0 --show

    # Ceiling RTSP camera, send to the IAC loopback, no window (lowest latency)
    python main.py --camera rtsp://admin:@192.168.1.15:554/stream1 --port "IAC Driver"

    # Try a mapping file with no MIDI hardware at all
    python main.py --dry-run --config config/default_mapping.json
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2

from src import __version__
from src.camera import CameraSource, CameraOpenError, parse_source
from src.gesture_features import FEATURE_DOCS, build_feature_vector
from src.hand_tracker import HandTracker
from src.midi_mapper import (
    ConfigError,
    ConfigWatcher,
    MidiMapper,
    build_smoother,
    load_config,
)
from src.midi_output import MidiUnavailableError, list_output_ports, open_output
from src.synth import GestureSynth
from src.visualizer import HudStats, Visualizer

log = logging.getLogger("gesture-midi")

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default_mapping.json"

HELP_KEYS = """\
keys:  q / ESC quit   m mute MIDI   p panic (all notes off)
       r reload mapping file   l toggle skeleton   d dump features"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-midi-controller",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_KEYS,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    camera = parser.add_argument_group("camera")
    camera.add_argument(
        "--camera", default="0",
        help="webcam index (0, 1, ...) or a stream URL such as "
             "rtsp://admin:@192.168.1.15:554/stream1 (default: 0)",
    )
    camera.add_argument("--width", type=int, default=None, help="requested capture width")
    camera.add_argument("--height", type=int, default=None, help="requested capture height")
    camera.add_argument("--fps", type=int, default=None, help="requested capture frame rate")
    camera.add_argument(
        "--no-mirror", dest="mirror", action="store_false",
        help="do not flip the image horizontally (mirror view is the default)",
    )
    camera.add_argument(
        "--rtsp-transport", choices=("tcp", "udp"), default="tcp",
        help="RTSP transport; udp is lower latency, tcp is more reliable (default: tcp)",
    )
    camera.add_argument(
        "--threaded-capture", dest="threaded", action="store_true", default=None,
        help="always drop stale frames in a grabber thread (automatic for streams)",
    )

    midi = parser.add_argument_group("MIDI")
    midi.add_argument(
        "--port", default=None,
        help="send to an existing MIDI port (name fragment or index) instead of "
             "creating a virtual one, e.g. --port 'IAC Driver'",
    )
    midi.add_argument("--port-name", default=None, help="name of the virtual MIDI port to create")
    midi.add_argument("--channel", type=int, default=None, help="override the MIDI channel (1-16)")
    midi.add_argument(
        "--dry-run", action="store_true",
        help="print MIDI messages instead of sending them (no MIDI backend needed)",
    )
    midi.add_argument("--list-ports", action="store_true", help="list MIDI output ports and exit")

    mapping = parser.add_argument_group("mapping")
    mapping.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"gesture -> MIDI mapping file (default: {DEFAULT_CONFIG.name})",
    )
    mapping.add_argument(
        "--no-hot-reload", dest="hot_reload", action="store_false",
        help="do not watch the mapping file for changes",
    )
    mapping.add_argument(
        "--list-features", action="store_true",
        help="list the feature names a mapping can use, and exit",
    )

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument("--max-hands", type=int, default=2, help="hands to track (default: 2)")
    tracking.add_argument(
        "--detection-confidence", type=float, default=0.5,
        help="MediaPipe minimum detection confidence (default: 0.5)",
    )
    tracking.add_argument(
        "--tracking-confidence", type=float, default=0.5,
        help="MediaPipe minimum tracking confidence (default: 0.5)",
    )
    tracking.add_argument(
        "--tracker-backend", choices=("auto", "tasks", "legacy"), default="auto",
        help="MediaPipe API to use (default: auto)",
    )
    tracking.add_argument(
        "--model", type=Path, default=None,
        help="path to hand_landmarker.task (downloaded and cached automatically)",
    )

    audio = parser.add_argument_group("built-in synth")
    audio.add_argument(
        "--synth", action="store_true",
        help="play the gestures through a built-in synth, so you can hear them "
             "without a DAW (holds a drone note for the filter to shape)",
    )
    audio.add_argument(
        "--drone", type=int, default=None, metavar="NOTE",
        help="MIDI note the synth holds continuously; -1 for none. Overrides the "
             "mapping file's 'synth' block (default: 45 for control patches, "
             "none for playable ones)",
    )
    audio.add_argument(
        "--synth-gain", type=float, default=None, help="synth output level (default: 0.28)",
    )

    display = parser.add_argument_group("display")
    display.add_argument("--show", action="store_true", help="open the preview window with the overlay")
    display.add_argument(
        "--debug-features", action="store_true",
        help="print the live feature vector to the console (useful for calibration)",
    )
    display.add_argument(
        "--stats-interval", type=float, default=5.0,
        help="seconds between throughput/latency reports, 0 to disable (default: 5)",
    )
    display.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="logging verbosity",
    )
    return parser


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
class GestureController:
    """Owns the capture -> tracking -> mapping -> MIDI pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config(args.config)
        if args.channel is not None:
            self._override_channel(args.channel)

        self.mapper = MidiMapper(self.config)
        self.smoother = build_smoother(self.config)
        self.watcher = ConfigWatcher(args.config) if args.hot_reload else None

        self.muted = False
        self.running = True
        self.synth: Optional[GestureSynth] = None
        self._frame_times: Deque[float] = deque(maxlen=60)
        self._latencies: Deque[float] = deque(maxlen=60)
        self._message_times: Deque[float] = deque(maxlen=200)
        self._last_stats = time.monotonic()
        self._frames = 0

    def _override_channel(self, channel: int) -> None:
        if not 1 <= channel <= 16:
            raise ConfigError(f"--channel must be between 1 and 16, got {channel}")
        self.config.midi.channel = channel
        for mapping in self.config.mappings:
            mapping.channel = channel

    # -- setup -------------------------------------------------------------
    def _open_output(self):
        port = self.args.port if self.args.port is not None else self.config.midi.port
        port_name = self.args.port_name or self.config.midi.port_name
        output = open_output(
            port=port, port_name=port_name, dry_run=self.args.dry_run,
            verbose=(self.args.log_level == "DEBUG"),
        )
        self.midi_target = getattr(output, "description", "dry run (no MIDI sent)")
        return output

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        args = self.args
        source = parse_source(args.camera)
        log.info("mapping file: %s", args.config)
        log.info("%s", self.config.describe())

        try:
            output = self._open_output()
        except (MidiUnavailableError, RuntimeError, ValueError) as exc:
            log.error("%s", exc)
            return 2

        if args.synth:
            # The patch chooses whether a drone makes sense; the flags still win.
            drone = self.config.synth.drone
            if args.drone is not None:
                drone = None if args.drone < 0 else args.drone
            try:
                self.synth = GestureSynth(
                    drone_note=drone,
                    gain=args.synth_gain if args.synth_gain is not None else self.config.synth.gain,
                ).start()
            except RuntimeError as exc:
                log.error("%s", exc)
                return 2

        visualizer = Visualizer() if args.show else None
        camera = CameraSource(
            source, width=args.width, height=args.height, fps=args.fps,
            mirror=args.mirror, threaded=args.threaded,
            rtsp_transport=args.rtsp_transport,
        )
        tracker = None
        try:
            camera.open()
            tracker = HandTracker(
                max_hands=args.max_hands,
                min_detection_confidence=args.detection_confidence,
                min_tracking_confidence=args.tracking_confidence,
                model_path=args.model,
                mirrored=args.mirror,
                backend=args.tracker_backend,
            )
            log.info("running — %s", HELP_KEYS.replace("\n", " | "))
            self._loop(camera, tracker, output, visualizer)
        except CameraOpenError as exc:
            log.error("%s", exc)
            return 2
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            for event in self.mapper.panic():
                try:
                    output.send(event)
                except Exception:  # pragma: no cover - best effort on shutdown
                    break
            if self.synth is not None:
                self.synth.stop()
            camera.close()
            if tracker is not None:
                tracker.close()
            output.close()
            if visualizer is not None:
                visualizer.close()
        return 0

    def _loop(self, camera, tracker, output, visualizer) -> None:
        args = self.args
        while self.running:
            frame = camera.read()
            if frame is None:
                log.warning("no more frames from %r, stopping", camera.source)
                break

            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            result = tracker.process(rgb, frame.timestamp_ms)
            hands = result.hands

            height, width = frame.image.shape[:2]
            aspect = width / height if height else 1.0
            raw_features = build_feature_vector(hands, self.config.calibration, aspect)
            features = self.smoother.update(raw_features)
            events = self.mapper.update(features, now=time.monotonic())

            if events and not self.muted:
                self._emit(events, output)
            now = time.monotonic()
            self._message_times.extend([now] * len(events))
            self._latencies.append((now - frame.timestamp) * 1000.0)
            self._frame_times.append(now)
            self._frames += 1

            if visualizer is not None:
                image = visualizer.render(frame.image, hands, self.mapper.states(), self._stats(len(result)))
                key = visualizer.show(image)
                self._handle_key(key, output, features, visualizer)
            if args.debug_features and self._frames % 15 == 0:
                self._print_features(features)

            self._maybe_reload(output)
            self._maybe_report()

    # -- helpers -----------------------------------------------------------
    def _emit(self, events: List, output) -> None:
        """Send events to the MIDI port and, if it is running, the built-in synth."""
        output.send_all(events)
        if self.synth is not None:
            self.synth.handle_all(events)

    def _stats(self, hands: int) -> HudStats:
        return HudStats(
            fps=self._fps(),
            latency_ms=(sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0,
            hands=hands,
            midi_target=self.midi_target,
            config_name=Path(self.args.config).name,
            muted=self.muted,
            messages_per_second=self._messages_per_second(),
        )

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _messages_per_second(self) -> float:
        if not self._message_times:
            return 0.0
        cutoff = time.monotonic() - 1.0
        while self._message_times and self._message_times[0] < cutoff:
            self._message_times.popleft()
        return float(len(self._message_times))

    def _handle_key(self, key: int, output, features, visualizer=None) -> None:
        if key in (-1, 255):
            return
        char = chr(key) if 0 <= key < 256 else ""
        if key == 27 or char == "q":
            self.running = False
        elif char == "m":
            self.muted = not self.muted
            if self.muted:
                self._emit(self.mapper.panic(), output)
            log.info("MIDI %s", "muted" if self.muted else "unmuted")
        elif char == "p":
            self._emit(self.mapper.panic(), output)
            log.info("panic sent")
        elif char == "r":
            self._reload(output)
        elif char == "l" and visualizer is not None:
            visualizer.show_landmarks = not visualizer.show_landmarks
        elif char == "d":
            self._print_features(features)

    def _print_features(self, features) -> None:
        if not features:
            print("no hands tracked", flush=True)
            return
        print(
            "  ".join(f"{name}={value:.2f}" for name, value in sorted(features.items())),
            flush=True,
        )

    def _maybe_reload(self, output) -> None:
        if self.watcher is None:
            return
        new_config = self.watcher.poll()
        if new_config is not None:
            self._apply_config(new_config, output)

    def _reload(self, output) -> None:
        try:
            new_config = load_config(self.args.config)
        except ConfigError as exc:
            log.error("reload failed: %s", exc)
            return
        self._apply_config(new_config, output)

    def _apply_config(self, new_config, output) -> None:
        # Release anything the old patch was holding before swapping it out.
        if not self.muted:
            self._emit(self.mapper.panic(), output)
        if self.args.channel is not None:
            self.config = new_config
            self._override_channel(self.args.channel)
        else:
            self.config = new_config
        self.mapper = MidiMapper(self.config)
        self.smoother = build_smoother(self.config)
        log.info("mapping reloaded: %s", self.config.describe().splitlines()[0])

    def _maybe_report(self) -> None:
        interval = self.args.stats_interval
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats < interval:
            return
        self._last_stats = now
        latency = (sum(self._latencies) / len(self._latencies)) if self._latencies else 0.0
        log.info(
            "%.1f fps | %.1f ms camera-to-MIDI | %.0f msg/s | %d frames",
            self._fps(), latency, self._messages_per_second(), self._frames,
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def print_features() -> None:
    print("Feature names usable in a mapping file's \"feature\" field:\n")
    width = max(len(name) for name in FEATURE_DOCS)
    for name, doc in FEATURE_DOCS.items():
        print(f"  {name:<{width}}  {doc}")
    print("\n<hand> is either 'left' or 'right', e.g. right.pinch")


def print_ports() -> int:
    try:
        ports = list_output_ports()
    except MidiUnavailableError as exc:
        print(exc)
        return 2
    if not ports:
        print("No MIDI output ports found.")
        print("On macOS: Audio MIDI Setup -> Window -> Show MIDI Studio -> IAC Driver -> "
              "tick 'Device is online'.")
        print("Or just run without --port to create a virtual port.")
        return 0
    print("MIDI output ports:")
    for index, name in enumerate(ports):
        print(f"  {index}: {name}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_features:
        print_features()
        return 0
    if args.list_ports:
        return print_ports()

    try:
        controller = GestureController(args)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    def _stop(_signum, _frame) -> None:
        controller.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
