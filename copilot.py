#!/usr/bin/env python3
"""AI-CoPilot: a real-time, webcam-driven driver safety dashboard.

The application combines MediaPipe FaceMesh (eye aspect ratio), DeepFace
(facial emotion recognition), OpenCV (rendering) and Pygame (alarm audio).
It also models a local-only emergency-dispatch workflow with a configurable
driver location. Everything runs on the laptop; no camera frame, location, or
incident record is transmitted, and no real emergency call is placed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.request
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# DeepFace can log emoji-laden status lines. Windows pipes/cp1252 consoles otherwise
# crash while DeepFace is downloading its first model.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import cv2
import mediapipe as mp
import numpy as np

try:
    import pygame
except Exception:  # Pygame is optional at import time; the alarm subsystem handles it.
    pygame = None


# ---------------------------------------------------------------------------
# Configuration and visual language
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
ALARM_FILE = APP_DIR / "assets" / "alarm.wav"
FACE_MODEL_FILE = APP_DIR / "assets" / "face_landmarker.task"
EMERGENCY_CONFIG_FILE = APP_DIR / "emergency_config.json"
INCIDENT_DIR = APP_DIR / "incidents"
INCIDENT_FILE = INCIDENT_DIR / "emergency_incidents.jsonl"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

DEFAULT_LOCATION_ADDRESS = (
    "Update emergency_config.json with your street address or landmark."
)
DEFAULT_EMERGENCY_CONFIG = {
    "driver_name": "Driver",
    "location": {
        "label": "Home",
        "address": DEFAULT_LOCATION_ADDRESS,
        "latitude": None,
        "longitude": None,
    },
    "dispatch_delay_seconds": 10,
}

WINDOW_NAME = "AI-CoPilot | Driver Safety System"
WINDOW_W, WINDOW_H = 1280, 720
CAMERA_W, DASHBOARD_W = 800, 480

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_DISPLAY = cv2.FONT_HERSHEY_DUPLEX

# All colours are OpenCV BGR values.
COLOR_BG = (32, 21, 13)
COLOR_BG_DEEP = (22, 14, 9)
COLOR_PANEL = (48, 33, 23)
COLOR_PANEL_RAISED = (57, 39, 28)
COLOR_BORDER = (76, 57, 43)
COLOR_TEXT = (247, 246, 243)
COLOR_MUTED = (169, 174, 183)
COLOR_DIM = (112, 119, 130)
COLOR_GREEN = (128, 222, 74)
COLOR_GREEN_DARK = (54, 93, 41)
COLOR_AMBER = (36, 191, 251)
COLOR_RED = (94, 77, 255)
COLOR_RED_DARK = (67, 35, 119)
COLOR_CYAN = (255, 214, 85)
COLOR_PURPLE = (250, 139, 167)

STRESS_EMOTIONS = {"angry", "fear", "disgust"}
RELAXING_EMOTIONS = {"sad", "angry", "fear", "disgust"}

_DASHBOARD_BASE: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
    font: int = FONT,
    align: str = "left",
) -> tuple[int, int]:
    """Draw text at a baseline position and return (width, height)."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if align == "center":
        x -= text_w // 2
    elif align == "right":
        x -= text_w
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return text_w, text_h


def measure_text(
    text: str,
    scale: float,
    thickness: int = 1,
    font: int = FONT,
) -> tuple[int, int]:
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    return text_w, text_h


def wrap_text(
    text: str,
    max_width: int,
    scale: float,
    thickness: int = 1,
    font: int = FONT,
) -> list[str]:
    """Greedy word wrap for OpenCV Hershey fonts."""
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if measure_text(candidate, scale, thickness, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def truncate_text(
    text: str,
    max_width: int,
    scale: float,
    thickness: int = 1,
    font: int = FONT,
) -> str:
    """Shorten a single line with an ellipsis so it fits max_width."""
    if measure_text(text, scale, thickness, font)[0] <= max_width:
        return text
    ellipsis = "..."
    while text and measure_text(text + ellipsis, scale, thickness, font)[0] > max_width:
        text = text[:-1]
    return text.rstrip() + ellipsis


def rounded_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int = 16,
    thickness: int = -1,
) -> None:
    """Draw a filled or outlined rectangle with rounded corners."""
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return
    radius = max(0, min(radius, width // 2, height // 2))
    if thickness < 0:
        if radius == 0:
            cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
            return
        cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), color, -1)
        cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), color, -1)
        cv2.circle(image, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x + width - radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x + radius, y + height - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(
            image,
            (x + width - radius, y + height - radius),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )
        return

    if radius == 0:
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness, cv2.LINE_AA)
        return
    cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(
        image,
        (x + radius, y + height),
        (x + width - radius, y + height),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(image, (x, y + radius), (x, y + height - radius), color, thickness, cv2.LINE_AA)
    cv2.line(
        image,
        (x + width, y + radius),
        (x + width, y + height - radius),
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x + radius, y + radius),
        (radius, radius),
        180,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x + width - radius, y + radius),
        (radius, radius),
        270,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x + width - radius, y + height - radius),
        (radius, radius),
        0,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (x + radius, y + height - radius),
        (radius, radius),
        90,
        0,
        90,
        color,
        thickness,
        cv2.LINE_AA,
    )


def blend_rounded_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
    radius: int = 16,
) -> None:
    """Alpha-blend a rounded rectangle onto an image."""
    alpha = max(0.0, min(1.0, alpha))
    overlay = image.copy()
    rounded_rect(overlay, rect, color, radius, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)


def draw_badge(
    image: np.ndarray,
    org: tuple[int, int],
    text: str,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    scale: float = 0.35,
    align: str = "left",
) -> tuple[int, int, int, int]:
    """Draw a compact pill-shaped label. org is its top-left/top-right anchor."""
    text_w, text_h = measure_text(text, scale, 1)
    pad_x, pad_y = 10, 6
    width = text_w + pad_x * 2
    height = text_h + pad_y * 2
    x, y = org
    if align == "center":
        x -= width // 2
    elif align == "right":
        x -= width
    rounded_rect(image, (x, y, width, height), bg, height // 2, -1)
    draw_text(image, text, (x + pad_x, y + pad_y + text_h), scale, fg, 1)
    return x, y, width, height


def draw_temperature(
    image: np.ndarray,
    x: int,
    baseline: int,
    value: str,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> int:
    """Draw a temperature with a vector degree mark and C, e.g. 22(deg)C."""
    width, text_h = measure_text(value, scale, thickness, FONT_DISPLAY)
    cv2.putText(
        image,
        value,
        (x, baseline),
        FONT_DISPLAY,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    dot_radius = max(2, int(round(scale * 3.0)))
    dot_x = x + width + 7
    dot_y = baseline - text_h + dot_radius
    cv2.circle(image, (dot_x, dot_y), dot_radius, color, thickness, cv2.LINE_AA)
    draw_text(
        image,
        "C",
        (dot_x + dot_radius + 5, baseline),
        scale * 0.43,
        color,
        max(1, thickness - 1),
        FONT_DISPLAY,
    )
    return width + dot_radius + 5 + int(scale * 10)


def draw_status_dot(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    now: float,
    radius: int = 6,
) -> None:
    pulse = (math.sin(now * 4.0) + 1.0) * 0.5
    cv2.circle(
        image,
        center,
        radius + int(round(4 * pulse)),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def draw_equaliser(
    image: np.ndarray,
    x: int,
    baseline: int,
    width: int,
    active: bool,
    color: tuple[int, int, int],
    now: float,
) -> None:
    bars = 11
    gap = 3
    bar_w = max(2, (width - gap * (bars - 1)) // bars)
    for index in range(bars):
        if active:
            phase = now * 8.2 + index * 0.91
            height = int(6 + 13 * (0.5 + 0.5 * math.sin(phase)))
        else:
            height = 3
        bx = x + index * (bar_w + gap)
        y1 = baseline - height
        cv2.rectangle(
            image,
            (bx, y1),
            (bx + bar_w, baseline),
            color if active else COLOR_DIM,
            -1,
        )


def safe_int(value: float) -> int:
    return int(round(max(0.0, value)))


# ---------------------------------------------------------------------------
# Audio alarm
# ---------------------------------------------------------------------------


def ensure_alarm_asset(path: Path) -> None:
    """Create a two-tone warning siren without requiring an external asset."""
    if path.exists() and path.stat().st_size > 1000:
        return

    sample_rate = 44_100
    segment_duration = 0.18
    segment_count = 10
    samples_per_segment = int(sample_rate * segment_duration)
    total_samples = samples_per_segment * segment_count
    audio = np.zeros(total_samples, dtype=np.float64)
    fade_samples = max(1, int(sample_rate * 0.012))

    for index in range(segment_count):
        start = index * samples_per_segment
        t = np.arange(samples_per_segment, dtype=np.float64) / sample_rate
        frequency = 980.0 if index % 2 == 0 else 735.0
        tone = (
            0.56 * np.sin(2.0 * math.pi * frequency * t)
            + 0.22 * np.sin(2.0 * math.pi * frequency * 2.0 * t)
        )
        envelope = np.ones(samples_per_segment, dtype=np.float64)
        envelope[:fade_samples] = np.linspace(0.0, 1.0, fade_samples)
        envelope[-fade_samples:] = np.linspace(1.0, 0.0, fade_samples)
        audio[start : start + samples_per_segment] = tone * envelope

    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0 * 0.9).astype("<i2")
    stereo = np.column_stack((pcm, pcm)).reshape(-1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo.tobytes())


class AlarmSystem:
    """Looping siren controlled by the drowsiness state machine."""

    def __init__(self, enabled: bool = True) -> None:
        self.available = False
        self.error: Optional[str] = None
        self._playing = False
        self._enabled = enabled
        self._sound = None

        if not enabled:
            self.error = "Disabled with --mute"
            return
        if pygame is None:
            self.error = "pygame is not installed"
            print("[audio] Pygame is unavailable; the visual alarm will still work.")
            return

        try:
            ensure_alarm_asset(ALARM_FILE)
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=1024)
            self._sound = pygame.mixer.Sound(str(ALARM_FILE))
            self._sound.set_volume(1.0)
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[audio] Alarm output unavailable: {self.error}")

    def start(self) -> None:
        if not self.available or self._sound is None or self._playing:
            return
        try:
            self._sound.play(loops=-1, fade_ms=80)
            self._playing = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[audio] Could not start the alarm: {self.error}")

    def stop(self) -> None:
        if not self._playing or self._sound is None:
            return
        try:
            self._sound.stop()
        except Exception:
            pass
        self._playing = False

    def close(self) -> None:
        self.stop()
        if pygame is not None:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Simulated emergency dispatch and location record
# ---------------------------------------------------------------------------


@dataclass
class EmergencyConfig:
    driver_name: str = "Driver"
    label: str = "Home"
    address: str = DEFAULT_LOCATION_ADDRESS
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    dispatch_delay_seconds: float = 10.0

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def is_configured(self) -> bool:
        address_set = bool(self.address.strip()) and self.address.strip() != DEFAULT_LOCATION_ADDRESS
        return self.has_coordinates or address_set

    @property
    def location_text(self) -> str:
        if not self.is_configured:
            return "Location not configured - edit emergency_config.json"
        parts: list[str] = []
        if self.label.strip():
            parts.append(self.label.strip())
        if self.address.strip() and self.address.strip() != DEFAULT_LOCATION_ADDRESS:
            parts.append(self.address.strip())
        if self.has_coordinates:
            parts.append(f"{self.latitude:.5f}, {self.longitude:.5f}")
        return " | ".join(parts)


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_emergency_config(path: Path = EMERGENCY_CONFIG_FILE) -> EmergencyConfig:
    """Load the local dispatch profile, creating a template on first use."""
    path = Path(path)
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(DEFAULT_EMERGENCY_CONFIG, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[dispatch] Created emergency profile: {path}")
            print("[dispatch] Add your address or coordinates to share a location.")
        except OSError as exc:
            print(f"[dispatch] Could not create {path}: {exc}")

    data: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
            else:
                raise ValueError("the top-level JSON value must be an object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[dispatch] Could not read {path}: {exc}. Using defaults.")

    location = data.get("location", {})
    if not isinstance(location, dict):
        location = {}

    latitude = _optional_float(location.get("latitude"))
    longitude = _optional_float(location.get("longitude"))
    if latitude is not None and not -90.0 <= latitude <= 90.0:
        print("[dispatch] Latitude is outside -90..90; ignoring coordinates.")
        latitude = None
    if longitude is not None and not -180.0 <= longitude <= 180.0:
        print("[dispatch] Longitude is outside -180..180; ignoring coordinates.")
        longitude = None
    if (latitude is None) != (longitude is None):
        print("[dispatch] Both latitude and longitude are required; ignoring coordinates.")
        latitude = longitude = None

    delay_value = _optional_float(data.get("dispatch_delay_seconds"))
    delay = 10.0 if delay_value is None else max(1.0, min(60.0, delay_value))

    driver_value = data.get("driver_name", "Driver")
    driver_name = str(driver_value).strip() if driver_value else "Driver"
    label_value = location.get("label", "Home")
    label = str(label_value).strip() if label_value else "Home"
    address_value = location.get("address", DEFAULT_LOCATION_ADDRESS)
    address = str(address_value).strip() if address_value else DEFAULT_LOCATION_ADDRESS

    return EmergencyConfig(
        driver_name=driver_name or "Driver",
        label=label or "Home",
        address=address or DEFAULT_LOCATION_ADDRESS,
        latitude=latitude,
        longitude=longitude,
        dispatch_delay_seconds=delay,
    )


@dataclass
class DispatchSnapshot:
    phase: str = "idle"
    remaining: float = 0.0
    delay: float = 10.0
    location: str = ""
    location_configured: bool = False
    case_id: str = ""


class EmergencyDispatcher:
    """Hands-free emergency-call simulation with a cancel window.

    This class deliberately has no network or telephone integration. It models the
    dispatch workflow, writes a local incident record, and makes the simulation state
    available to the dashboard.
    """

    def __init__(
        self,
        config_path: Path = EMERGENCY_CONFIG_FILE,
        incident_file: Path = INCIDENT_FILE,
    ) -> None:
        self._config_path = Path(config_path)
        self._incident_file = Path(incident_file)
        self._config = load_emergency_config(self._config_path)
        self._phase = "idle"
        self._deadline = 0.0
        self._remaining = 0.0
        self._delay = self._config.dispatch_delay_seconds
        self._location = self._config.location_text
        self._location_configured = self._config.is_configured
        self._case_id = ""

        if not self._config.is_configured:
            print(
                "[dispatch] Emergency location is not configured. "
                f"Edit {self._config_path} to attach an address."
            )

    @property
    def phase(self) -> str:
        return self._phase

    def arm(self, now: float) -> None:
        """Start the countdown after a real drowsiness event."""
        self._config = load_emergency_config(self._config_path)
        self._delay = self._config.dispatch_delay_seconds
        self._location = self._config.location_text
        self._location_configured = self._config.is_configured
        self._phase = "countdown"
        self._deadline = now + self._delay
        self._remaining = self._delay
        self._case_id = ""
        print(
            "[dispatch] Simulated emergency call armed. "
            f"Dispatch in {self._delay:.0f}s; press C to cancel."
        )

    def update(self, now: float) -> DispatchSnapshot:
        if self._phase == "countdown":
            self._remaining = max(0.0, self._deadline - now)
            if self._remaining <= 0.0:
                self._dispatch()
        return self.snapshot(now)

    def cancel(self) -> bool:
        """Cancel a pending simulated call. Returns True if it was still pending."""
        if self._phase != "countdown":
            return False
        self._phase = "cancelled"
        self._remaining = 0.0
        print("[dispatch] Emergency dispatch cancelled by the driver.")
        return True

    def reset(self) -> None:
        self._phase = "idle"
        self._deadline = 0.0
        self._remaining = 0.0
        self._case_id = ""

    def snapshot(self, now: float) -> DispatchSnapshot:
        remaining = max(0.0, self._deadline - now) if self._phase == "countdown" else 0.0
        return DispatchSnapshot(
            phase=self._phase,
            remaining=remaining,
            delay=self._delay,
            location=self._location,
            location_configured=self._location_configured,
            case_id=self._case_id,
        )

    def _dispatch(self) -> None:
        self._phase = "dispatched"
        self._remaining = 0.0
        timestamp = datetime.now().astimezone()
        self._case_id = f"EC-{timestamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4].upper()}"

        record = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "case_id": self._case_id,
            "driver": self._config.driver_name,
            "reason": "Prolonged eye closure / drowsiness detected",
            "location": self._location,
            "latitude": self._config.latitude,
            "longitude": self._config.longitude,
            "location_configured": self._location_configured,
            "vehicle_speed_kmh": 0,
            "dispatch_mode": "simulated",
            "real_emergency_call_placed": False,
        }
        try:
            self._incident_file.parent.mkdir(parents=True, exist_ok=True)
            with self._incident_file.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"[dispatch] SIMULATED emergency dispatch {self._case_id}. "
                f"Location: {self._location}. Log: {self._incident_file}"
            )
        except OSError as exc:
            print(f"[dispatch] Could not write incident log: {exc}")


# ---------------------------------------------------------------------------
# FaceMesh and drowsiness detection
# ---------------------------------------------------------------------------


@dataclass
class FaceReading:
    found: bool = False
    ear: Optional[float] = None
    box: Optional[tuple[int, int, int, int]] = None


def ensure_face_model(path: Path = FACE_MODEL_FILE) -> None:
    """Fetch the modern MediaPipe Face Landmarker task bundle if it is absent."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    print("[vision] Downloading the MediaPipe Face Landmarker model (one time)...")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded model file is unexpectedly small")
        temporary.replace(path)
        print(f"[vision] Model saved to {path}")
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            "The MediaPipe Face Landmarker model is missing and could not be downloaded. "
            "Connect to the internet for the first run or place face_landmarker.task in assets/."
        ) from exc


class FaceMonitor:
    """MediaPipe Face Landmarker wrapper that computes Eye Aspect Ratio (EAR)."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        ensure_face_model()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_FILE)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray) -> float:
        vertical_1 = float(np.linalg.norm(points[1] - points[5]))
        vertical_2 = float(np.linalg.norm(points[2] - points[4]))
        horizontal = float(np.linalg.norm(points[0] - points[3]))
        if horizontal <= 1e-6:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def process(self, frame_bgr: np.ndarray, now: float) -> FaceReading:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(now * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return FaceReading(found=False)

        landmarks = result.face_landmarks[0]
        all_points = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)
        left_points = all_points[list(self.LEFT_EYE)]
        right_points = all_points[list(self.RIGHT_EYE)]
        ear = (self._eye_aspect_ratio(left_points) + self._eye_aspect_ratio(right_points)) / 2.0

        xs = np.clip(all_points[:, 0] * width, 0, width - 1)
        ys = np.clip(all_points[:, 1] * height, 0, height - 1)
        box_w = float(xs.max() - xs.min())
        box_h = float(ys.max() - ys.min())
        pad_x = int(box_w * 0.13)
        pad_y = int(box_h * 0.13)
        x1 = max(0, int(xs.min()) - pad_x)
        y1 = max(0, int(ys.min()) - pad_y)
        x2 = min(width - 1, int(xs.max()) + pad_x)
        y2 = min(height - 1, int(ys.max()) + pad_y)

        return FaceReading(found=True, ear=float(ear), box=(x1, y1, x2, y2))

    def close(self) -> None:
        self._landmarker.close()


@dataclass
class DrowsinessReading:
    ear: Optional[float] = None
    closed_seconds: float = 0.0
    drowsy: bool = False
    trigger_alarm: bool = False


class DrowsinessMonitor:
    """Temporal EAR filter and the 2.5-second microsleep state machine."""

    def __init__(
        self,
        ear_threshold: float = 0.21,
        drowsy_seconds: float = 2.5,
        smoothing_frames: int = 3,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.drowsy_seconds = drowsy_seconds
        self._history: deque[float] = deque(maxlen=max(1, smoothing_frames))
        self._closed_since: Optional[float] = None
        self._alarm_latched = False

    def update(
        self,
        face_found: bool,
        ear: Optional[float],
        now: float,
    ) -> DrowsinessReading:
        if not face_found or ear is None:
            self._history.clear()
            self._closed_since = None
            return DrowsinessReading(ear=None, closed_seconds=0.0)

        self._history.append(float(ear))
        smoothed_ear = float(np.median(np.asarray(self._history, dtype=np.float64)))
        eyes_closed = smoothed_ear < self.ear_threshold

        if eyes_closed:
            if self._closed_since is None:
                self._closed_since = now
            closed_seconds = max(0.0, now - self._closed_since)
        else:
            self._closed_since = None
            closed_seconds = 0.0

        trigger = (
            eyes_closed
            and closed_seconds >= self.drowsy_seconds
            and not self._alarm_latched
        )
        if trigger:
            self._alarm_latched = True

        return DrowsinessReading(
            ear=smoothed_ear,
            closed_seconds=closed_seconds,
            drowsy=eyes_closed and closed_seconds >= 0.4,
            trigger_alarm=trigger,
        )

    def reset(self) -> None:
        self._history.clear()
        self._closed_since = None
        self._alarm_latched = False


# ---------------------------------------------------------------------------
# Asynchronous emotion recognition
# ---------------------------------------------------------------------------


@dataclass
class EmotionSnapshot:
    emotion: Optional[str] = None
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    status: str = "loading"
    error: Optional[str] = None
    updated_at: float = 0.0


class EmotionAnalyzer:
    """Single-worker DeepFace pipeline that never blocks the render loop."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None
        self._snapshot = EmotionSnapshot(
            status="loading" if enabled else "disabled",
            error=None if enabled else "Emotion analysis disabled",
        )
        self._thread: Optional[threading.Thread] = None

        if enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="ai-copilot-emotion",
                daemon=True,
            )
            self._thread.start()

    def submit(self, face_crop: np.ndarray) -> None:
        if not self.enabled or self._stop_event.is_set():
            return
        with self._lock:
            if not self.enabled:
                return
            self._pending_frame = face_crop.copy()
            self._snapshot.status = "analyzing"
            self._snapshot.error = None
            self._wake.clear()
            self._wake.set()

    def snapshot(self) -> EmotionSnapshot:
        with self._lock:
            return EmotionSnapshot(
                emotion=self._snapshot.emotion,
                confidence=self._snapshot.confidence,
                scores=dict(self._snapshot.scores),
                status=self._snapshot.status,
                error=self._snapshot.error,
                updated_at=self._snapshot.updated_at,
            )

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot.status = "error"
            self._snapshot.error = message[:220]

    def _run(self) -> None:
        try:
            from deepface import DeepFace
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print(f"[emotion] DeepFace could not be loaded: {detail}")
            traceback.print_exc()
            with self._lock:
                self.enabled = False
                self._snapshot.status = "error"
                self._snapshot.error = detail[:220]
            return

        with self._lock:
            self._snapshot.status = "ready"
            self._snapshot.error = None
        print("[emotion] DeepFace is ready. The first scan may download model weights.")

        while not self._stop_event.is_set():
            if not self._wake.wait(timeout=0.5):
                continue
            if self._stop_event.is_set():
                break

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._wake.clear()
                if frame is not None:
                    self._snapshot.status = "analyzing"

            if frame is None:
                continue

            try:
                # MediaPipe has already isolated the driver's face, so DeepFace can
                # skip a second detector and run the emotion network directly.
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="skip",
                    enforce_detection=False,
                    align=False,
                    silent=True,
                )
                if isinstance(result, (list, tuple)):
                    if not result:
                        raise ValueError("DeepFace returned no analysis result")
                    result = result[0]
                emotion = str(result.get("dominant_emotion", "neutral")).lower()
                raw_scores = result.get("emotion", {}) or {}
                scores = {
                    str(name).lower(): float(value)
                    for name, value in raw_scores.items()
                }
                confidence = max(0.0, min(1.0, scores.get(emotion, 0.0) / 100.0))

                with self._lock:
                    self._snapshot.emotion = emotion
                    self._snapshot.confidence = confidence
                    self._snapshot.scores = scores
                    self._snapshot.status = "ready"
                    self._snapshot.error = None
                    self._snapshot.updated_at = time.monotonic()
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                print(f"[emotion] Analysis failed: {detail}")
                traceback.print_exc()
                self._set_error(detail)
                if self._stop_event.wait(timeout=3.0):
                    break

    def close(self) -> None:
        self._stop_event.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)


# ---------------------------------------------------------------------------
# UI state and layout
# ---------------------------------------------------------------------------


@dataclass
class DashboardState:
    speed_kmh: float = 0.0
    ac_temp_c: float = 22.0
    music: str = "Standby"
    status: str = "Safe"
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    emotion_status: str = "loading"
    emotion_error: Optional[str] = None
    ear: Optional[float] = None
    closed_seconds: float = 0.0
    emergency: bool = False
    face_found: bool = False
    fps: float = 0.0
    stress_response: bool = False
    alarm_available: bool = True
    ear_threshold: float = 0.21
    drowsy_seconds: float = 2.5
    dispatch_phase: str = "idle"
    dispatch_remaining: float = 0.0
    dispatch_delay: float = 10.0
    dispatch_location: str = ""
    dispatch_location_configured: bool = False
    dispatch_case_id: str = ""


def state_colour(state: DashboardState) -> tuple[int, int, int]:
    if state.emergency:
        return COLOR_RED
    if state.status == "Drowsy":
        return COLOR_AMBER
    if state.status == "No Driver":
        return COLOR_DIM
    return COLOR_GREEN


def _dashboard_base() -> np.ndarray:
    global _DASHBOARD_BASE
    if _DASHBOARD_BASE is None:
        top = np.array(COLOR_BG, dtype=np.float32)
        bottom = np.array((46, 30, 19), dtype=np.float32)
        gradient = np.linspace(0.0, 1.0, WINDOW_H, dtype=np.float32)[:, None, None]
        background = top[None, None, :] * (1.0 - gradient) + bottom[None, None, :] * gradient
        background = np.repeat(background, DASHBOARD_W, axis=1).astype(np.uint8)
        for y in range(0, WINDOW_H, 36):
            cv2.line(background, (0, y), (DASHBOARD_W, y), (39, 27, 19), 1)
        for x in range(0, DASHBOARD_W, 36):
            cv2.line(background, (x, 0), (x, WINDOW_H), (39, 27, 19), 1)
        _DASHBOARD_BASE = background
    return _DASHBOARD_BASE.copy()


def _camera_base() -> np.ndarray:
    panel = np.full((WINDOW_H, CAMERA_W, 3), COLOR_BG_DEEP, dtype=np.uint8)
    for x in range(0, CAMERA_W, 44):
        cv2.line(panel, (x, 0), (x, WINDOW_H), (27, 19, 13), 1)
    for y in range(0, WINDOW_H, 44):
        cv2.line(panel, (0, y), (CAMERA_W, y), (27, 19, 13), 1)
    return panel


def _draw_camera_header(panel: np.ndarray, state: DashboardState, now: float) -> None:
    cv2.rectangle(panel, (0, 0), (CAMERA_W, 63), COLOR_BG_DEEP, -1)
    cv2.line(panel, (0, 63), (CAMERA_W, 63), COLOR_BORDER, 1)
    draw_text(panel, "DRIVER CAMERA", (24, 27), 0.58, COLOR_TEXT, 2, FONT_DISPLAY)
    draw_text(panel, "LIVE COMPUTER VISION FEED", (24, 49), 0.34, COLOR_MUTED, 1)

    if state.face_found:
        draw_status_dot(panel, (551, 24), COLOR_GREEN, now, 4)
        draw_text(panel, "FACE LOCK", (564, 29), 0.36, COLOR_GREEN, 1)
    else:
        draw_status_dot(panel, (541, 24), COLOR_AMBER, now, 4)
        draw_text(panel, "SEARCHING", (554, 29), 0.36, COLOR_AMBER, 1)

    draw_text(
        panel,
        f"{state.fps:4.1f} FPS",
        (CAMERA_W - 24, 29),
        0.4,
        COLOR_MUTED,
        1,
        align="right",
    )
    draw_text(panel, "CAM 01  |  640 x 480", (CAMERA_W - 24, 49), 0.32, COLOR_DIM, 1, align="right")
    draw_badge(
        panel,
        (23, 69),
        "AI VISION ONLINE" if state.face_found else "AI VISION STANDBY",
        COLOR_GREEN if state.face_found else COLOR_MUTED,
        (30, 48, 29) if state.face_found else (37, 30, 24),
        0.31,
    )


def _draw_face_tracking(
    panel: np.ndarray,
    face_box: tuple[int, int, int, int],
    scale: float,
    offset: tuple[int, int],
    state: DashboardState,
    now: float,
) -> None:
    x1, y1, x2, y2 = face_box
    offset_x, offset_y = offset
    left = int(offset_x + x1 * scale)
    top = int(offset_y + y1 * scale)
    right = int(offset_x + x2 * scale)
    bottom = int(offset_y + y2 * scale)
    if right <= left or bottom <= top:
        return

    colour = state_colour(state)
    cv2.rectangle(panel, (left, top), (right, bottom), colour, 1, cv2.LINE_AA)
    corner = max(14, min(32, (right - left) // 5, (bottom - top) // 5))
    thickness = 4 if state.status != "Safe" else 3
    segments = (
        ((left, top), (left + corner, top)),
        ((left, top), (left, top + corner)),
        ((right, top), (right - corner, top)),
        ((right, top), (right, top + corner)),
        ((left, bottom), (left + corner, bottom)),
        ((left, bottom), (left, bottom - corner)),
        ((right, bottom), (right - corner, bottom)),
        ((right, bottom), (right, bottom - corner)),
    )
    for start, end in segments:
        cv2.line(panel, start, end, colour, thickness, cv2.LINE_AA)

    if state.emergency:
        label = "DROWSINESS ALERT"
    elif state.status == "Drowsy":
        label = "DRIVER  |  EYES CLOSING"
    elif state.status == "No Driver":
        label = "DRIVER  |  NO FACE"
    else:
        label = "DRIVER  |  ATTENTIVE"
    label_scale = 0.4
    label_w, label_h = measure_text(label, label_scale, 1)
    badge_w, badge_h = label_w + 20, label_h + 12
    badge_x = left
    badge_y = top - badge_h - 8
    if badge_y < 82:
        badge_y = top + 8
    if badge_x + badge_w > CAMERA_W - 12:
        badge_x = CAMERA_W - 12 - badge_w
    rounded_rect(panel, (badge_x, badge_y, badge_w, badge_h), colour, 6, -1)
    draw_text(
        panel,
        label,
        (badge_x + 10, badge_y + 6 + label_h),
        label_scale,
        COLOR_BG_DEEP,
        1,
    )
    draw_status_dot(panel, (right - 13, top + 15), colour, now, 3)


def _draw_camera_metrics(
    panel: np.ndarray,
    state: DashboardState,
    now: float,
) -> None:
    if state.emergency:
        cv2.rectangle(panel, (0, 618), (CAMERA_W, WINDOW_H), COLOR_BG_DEEP, -1)
        flash = (math.sin(now * 2.0 * math.pi * 2.4) + 1.0) * 0.5
        banner = (35, 22, 140 + int(70 * flash))
        cv2.rectangle(panel, (0, 618), (CAMERA_W, WINDOW_H), banner, -1)
        draw_text(
            panel,
            "DROWSINESS DETECTED - PULLING OVER",
            (CAMERA_W // 2, 648),
            0.58,
            COLOR_TEXT,
            2,
            FONT_DISPLAY,
            align="center",
        )
        if state.dispatch_phase == "countdown":
            dispatch_line = (
                f"SIMULATED CALL IN {state.dispatch_remaining:04.1f}s"
                "  |  C cancel  |  R reset"
            )
        elif state.dispatch_phase == "dispatched":
            dispatch_line = "SIMULATED DISPATCH SENT  |  R reset"
        elif state.dispatch_phase == "cancelled":
            dispatch_line = "DISPATCH CANCELLED  |  ALARM ACTIVE  |  R reset"
        else:
            dispatch_line = "Emergency stop engaged  |  Press R to reset"
        draw_text(
            panel,
            dispatch_line,
            (CAMERA_W // 2, 677),
            0.35,
            (222, 214, 218),
            1,
            align="center",
        )
        location_text = truncate_text(
            f"LOCATION: {state.dispatch_location or 'not configured'}",
            CAMERA_W - 70,
            0.3,
        )
        draw_text(
            panel,
            location_text,
            (CAMERA_W // 2, 703),
            0.3,
            COLOR_TEXT if state.dispatch_location_configured else COLOR_AMBER,
            1,
            align="center",
        )
        return

    metrics_h = 72
    top = WINDOW_H - metrics_h
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, top), (CAMERA_W, WINDOW_H), (10, 11, 13), -1)
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)
    cv2.line(panel, (0, top), (CAMERA_W, top), (85, 84, 82), 1)

    face_value = "LOCKED" if state.face_found else "SEARCHING"
    face_colour = COLOR_GREEN if state.face_found else COLOR_AMBER
    ear_value = f"{state.ear:.3f}" if state.ear is not None else "--"
    closure_value = f"{state.closed_seconds:.1f} / {state.drowsy_seconds:.1f}s"
    closure_colour = COLOR_AMBER if state.closed_seconds >= 0.4 else COLOR_TEXT

    columns = (
        (26, "FACE TRACK", face_value, face_colour),
        (238, "EYE ASPECT RATIO", ear_value, COLOR_CYAN),
        (455, "EYE CLOSURE", closure_value, closure_colour),
        (655, "SENSOR RATE", f"{state.fps:4.1f} FPS", COLOR_TEXT),
    )
    for x, label, value, colour in columns:
        draw_text(panel, label, (x, top + 23), 0.31, COLOR_MUTED, 1)
        draw_text(panel, value, (x, top + 52), 0.53, colour, 1, FONT_DISPLAY)


def render_camera_panel(
    frame: Optional[np.ndarray],
    state: DashboardState,
    face_box: Optional[tuple[int, int, int, int]],
    now: float,
) -> np.ndarray:
    panel = _camera_base()

    if frame is not None:
        frame_h, frame_w = frame.shape[:2]
        area_top = 64
        available_h = WINDOW_H - area_top
        scale = min(CAMERA_W / frame_w, available_h / frame_h)
        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))
        offset_x = (CAMERA_W - new_w) // 2
        offset_y = area_top + (available_h - new_h) // 2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        panel[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        if face_box is not None and state.face_found:
            _draw_face_tracking(panel, face_box, scale, (offset_x, offset_y), state, now)
        cv2.line(panel, (0, 63), (CAMERA_W, 63), COLOR_BORDER, 1)
    else:
        centre_y = WINDOW_H // 2
        draw_text(
            panel,
            "CAMERA OFFLINE",
            (CAMERA_W // 2, centre_y - 20),
            1.1,
            COLOR_TEXT,
            2,
            FONT_DISPLAY,
            align="center",
        )
        draw_text(
            panel,
            "Allow camera access or try --camera 1",
            (CAMERA_W // 2, centre_y + 18),
            0.45,
            COLOR_MUTED,
            1,
            align="center",
        )

    _draw_camera_header(panel, state, now)
    _draw_camera_metrics(panel, state, now)

    if state.emergency:
        flash = (math.sin(now * 2.0 * math.pi * 2.4) + 1.0) * 0.5
        colour = COLOR_RED if flash > 0.45 else COLOR_RED_DARK
        cv2.rectangle(panel, (2, 64), (CAMERA_W - 3, WINDOW_H - 3), colour, 5)
    return panel


def render_dashboard(state: DashboardState, now: float) -> np.ndarray:
    if state.emergency:
        return _render_emergency_dashboard(state, now)

    canvas = _dashboard_base()
    margin, content_w = 24, DASHBOARD_W - 48
    accent = state_colour(state)

    # Header
    draw_text(canvas, "AI-CO PILOT", (margin, 44), 0.78, COLOR_TEXT, 2, FONT_DISPLAY)
    draw_text(
        canvas,
        "REAL-TIME DRIVER SAFETY SYSTEM",
        (margin, 69),
        0.33,
        COLOR_MUTED,
        1,
    )
    draw_badge(
        canvas,
        (DASHBOARD_W - margin, 23),
        "SYSTEM ARMED",
        COLOR_GREEN,
        (29, 50, 30),
        0.31,
        align="right",
    )
    cv2.line(
        canvas,
        (margin, 86),
        (DASHBOARD_W - margin, 86),
        COLOR_BORDER,
        1,
    )

    # Speed card
    rounded_rect(canvas, (margin, 104, content_w, 150), COLOR_PANEL, 18, -1)
    rounded_rect(canvas, (margin, 104, content_w, 150), COLOR_BORDER, 18, 1)
    draw_text(canvas, "VEHICLE SPEED", (margin + 20, 132), 0.36, COLOR_MUTED, 1)
    speed_text = f"{safe_int(state.speed_kmh)}"
    speed_w, _ = draw_text(
        canvas,
        speed_text,
        (margin + 16, 219),
        2.55,
        COLOR_TEXT,
        4,
        FONT_DISPLAY,
    )
    draw_text(
        canvas,
        "km/h",
        (margin + 26 + speed_w, 211),
        0.58,
        COLOR_MUTED,
        1,
    )
    draw_text(
        canvas,
        "CRUISE CONTROL",
        (DASHBOARD_W - margin - 20, 132),
        0.32,
        COLOR_MUTED,
        1,
        align="right",
    )
    draw_status_dot(canvas, (DASHBOARD_W - margin - 20 - 72, 151), COLOR_GREEN, now, 3)
    draw_text(
        canvas,
        "ACTIVE",
        (DASHBOARD_W - margin - 20, 156),
        0.39,
        COLOR_GREEN,
        1,
        align="right",
    )
    track_x1, track_x2 = margin + 20, DASHBOARD_W - margin - 20
    cv2.rectangle(canvas, (track_x1, 234), (track_x2, 240), (72, 54, 43), -1)
    speed_ratio = min(1.0, max(0.0, state.speed_kmh / 120.0))
    fill_x = track_x1 + int((track_x2 - track_x1) * speed_ratio)
    cv2.rectangle(canvas, (track_x1, 234), (max(track_x1, fill_x), 240), COLOR_CYAN, -1)
    draw_text(canvas, "0", (track_x1, 251), 0.28, COLOR_DIM, 1)
    draw_text(canvas, "120", (track_x2, 251), 0.28, COLOR_DIM, 1, align="right")

    # AC card
    ac_rect = (margin, 270, 208, 134)
    rounded_rect(canvas, ac_rect, COLOR_PANEL, 16, -1)
    rounded_rect(canvas, ac_rect, COLOR_BORDER, 16, 1)
    draw_text(canvas, "SMART AC", (margin + 18, 297), 0.35, COLOR_MUTED, 1)
    ac_colour = COLOR_CYAN if state.stress_response else COLOR_TEXT
    draw_temperature(
        canvas,
        margin + 16,
        363,
        str(safe_int(state.ac_temp_c)),
        1.65,
        ac_colour,
        2,
    )
    if state.stress_response:
        draw_text(canvas, "Cooling boost -2 C", (margin + 18, 389), 0.34, COLOR_CYAN, 1)
    else:
        draw_text(canvas, f"AC Temp: {safe_int(state.ac_temp_c)} C", (margin + 18, 389), 0.34, COLOR_DIM, 1)

    # Music card
    music_x = margin + 224
    music_rect = (music_x, 270, 208, 134)
    rounded_rect(canvas, music_rect, COLOR_PANEL, 16, -1)
    rounded_rect(canvas, music_rect, COLOR_BORDER, 16, 1)
    draw_text(canvas, "MUSIC", (music_x + 18, 297), 0.35, COLOR_MUTED, 1)
    music_active = "playing" in state.music.lower()
    draw_status_dot(
        canvas,
        (music_x + 183, 290),
        COLOR_PURPLE if music_active else COLOR_DIM,
        now,
        3,
    )
    lines = wrap_text(state.music, 172, 0.41, 1)
    for index, line in enumerate(lines[:2]):
        draw_text(
            canvas,
            line,
            (music_x + 18, 331 + index * 22),
            0.41,
            COLOR_TEXT if music_active else COLOR_MUTED,
            1,
        )
    draw_equaliser(
        canvas,
        music_x + 18,
        388,
        106,
        music_active,
        COLOR_PURPLE,
        now,
    )
    draw_text(canvas, "CABIN AUDIO", (music_x + 190, 389), 0.26, COLOR_DIM, 1, align="right")

    # Safety status card
    status_rect = (margin, 420, content_w, 116)
    rounded_rect(canvas, status_rect, COLOR_PANEL, 16, -1)
    blend_rounded_rect(canvas, status_rect, accent, 0.11, 16)
    rounded_rect(canvas, status_rect, accent, 16, 1)
    rounded_rect(canvas, (margin, 420, 5, 116), accent, 2, -1)
    draw_text(canvas, "SAFETY STATUS", (margin + 20, 447), 0.34, COLOR_MUTED, 1)
    draw_text(
        canvas,
        state.status,
        (margin + 18, 506),
        1.02,
        accent,
        2,
        FONT_DISPLAY,
    )
    if state.status == "Drowsy":
        detail = f"Eyes closed {state.closed_seconds:.1f}s / {state.drowsy_seconds:.1f}s"
        detail_colour = COLOR_AMBER
    elif state.status == "No Driver":
        detail = "Waiting for a clear view of the driver"
        detail_colour = COLOR_MUTED
    elif state.ear is not None:
        detail = f"Eyes open  |  EAR {state.ear:.3f}"
        detail_colour = COLOR_MUTED
    else:
        detail = "Monitoring face geometry"
        detail_colour = COLOR_MUTED
    draw_text(canvas, detail, (margin + 20, 526), 0.34, detail_colour, 1)
    draw_status_dot(
        canvas,
        (DASHBOARD_W - margin - 24, 456),
        accent,
        now,
        7,
    )
    if state.status == "Drowsy":
        draw_badge(
            canvas,
            (DASHBOARD_W - margin - 20, 489),
            "MICROSLEEP RISK",
            COLOR_BG_DEEP,
            COLOR_AMBER,
            0.29,
            align="right",
        )

    # Driver emotion card
    mood_rect = (margin, 550, content_w, 84)
    rounded_rect(canvas, mood_rect, COLOR_PANEL, 16, -1)
    rounded_rect(canvas, mood_rect, COLOR_BORDER, 16, 1)
    draw_text(canvas, "DRIVER EMOTION", (margin + 18, 574), 0.32, COLOR_MUTED, 1)
    draw_text(canvas, "SMART CABIN", (DASHBOARD_W - margin - 18, 574), 0.32, COLOR_MUTED, 1, align="right")

    if state.emotion_status == "disabled":
        mood_text = "Disabled"
        mood_colour = COLOR_DIM
    elif state.emotion_status == "error":
        mood_text = "AI unavailable"
        mood_colour = COLOR_RED
    elif state.emotion_status in {"loading", "analyzing"} and state.emotion is None:
        mood_text = "Calibrating..."
        mood_colour = COLOR_MUTED
    elif state.emotion is None:
        mood_text = "No face sample"
        mood_colour = COLOR_MUTED
    else:
        mood_text = f"{state.emotion.title()}  {int(round(state.emotion_confidence * 100))}%"
        mood_colour = COLOR_PURPLE if not state.stress_response else COLOR_AMBER

    draw_text(canvas, mood_text, (margin + 18, 606), 0.7, mood_colour, 1, FONT_DISPLAY)
    bar_x, bar_y, bar_w = margin + 18, 616, 170
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (69, 55, 44), -1)
    if state.emotion is not None:
        conf_w = int(bar_w * max(0.0, min(1.0, state.emotion_confidence)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + conf_w, bar_y + 5), mood_colour, -1)

    if state.emotion_status in {"loading", "analyzing"} and state.emotion is None:
        response = "DEEP LEARNING"
        response_colour = COLOR_MUTED
    elif state.emotion_status == "error":
        response = "FALLBACK ACTIVE"
        response_colour = COLOR_RED
    elif state.emotion_status == "disabled":
        response = "MANUAL CLIMATE"
        response_colour = COLOR_DIM
    elif state.stress_response:
        response = "COOLING +2 C"
        response_colour = COLOR_CYAN
    else:
        response = "COMFORT AUTO"
        response_colour = COLOR_GREEN
    draw_text(
        canvas,
        response,
        (DASHBOARD_W - margin - 18, 606),
        0.38,
        response_colour,
        1,
        align="right",
    )

    # Footer
    cv2.line(canvas, (margin, 654), (DASHBOARD_W - margin, 654), COLOR_BORDER, 1)
    draw_text(canvas, "Q  Quit", (margin, 681), 0.38, COLOR_MUTED, 1)
    draw_text(canvas, "R  Reset", (margin + 94, 681), 0.38, COLOR_MUTED, 1)
    draw_text(
        canvas,
        "SOS AUDIO READY" if state.alarm_available else "SOS AUDIO OFFLINE",
        (DASHBOARD_W - margin, 681),
        0.3,
        COLOR_DIM if state.alarm_available else COLOR_RED,
        1,
        align="right",
    )
    draw_text(
        canvas,
        "SAFETY SIMULATION - NOT FOR VEHICLE CONTROL",
        (DASHBOARD_W // 2, 706),
        0.27,
        COLOR_DIM,
        1,
        align="center",
    )
    return canvas


def _render_emergency_dashboard(state: DashboardState, now: float) -> np.ndarray:
    canvas = _dashboard_base()
    flash = (math.sin(now * 2.0 * math.pi * 2.4) + 1.0) * 0.5
    overlay = canvas.copy()
    overlay[:] = (24, 20, int(120 + 90 * flash))
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    border_colour = COLOR_RED if flash > 0.35 else COLOR_RED_DARK
    cv2.rectangle(canvas, (3, 3), (DASHBOARD_W - 4, WINDOW_H - 4), border_colour, 7)

    draw_text(canvas, "AI-CO PILOT", (24, 43), 0.67, COLOR_TEXT, 2, FONT_DISPLAY)
    draw_badge(
        canvas,
        (DASHBOARD_W - 24, 22),
        "SOS / ALERT",
        COLOR_TEXT,
        (55, 34, 130),
        0.34,
        align="right",
    )

    # Warning triangle
    triangle_colour = COLOR_TEXT if flash > 0.4 else COLOR_AMBER
    pts = np.array([[DASHBOARD_W // 2, 88], [DASHBOARD_W // 2 + 48, 173], [DASHBOARD_W // 2 - 48, 173]])
    cv2.polylines(canvas, [pts], True, triangle_colour, 4, cv2.LINE_AA)
    draw_text(
        canvas,
        "!",
        (DASHBOARD_W // 2, 159),
        1.25,
        triangle_colour,
        3,
        FONT_DISPLAY,
        align="center",
    )

    draw_text(
        canvas,
        "DROWSINESS DETECTED",
        (DASHBOARD_W // 2, 231),
        0.8,
        COLOR_TEXT,
        2,
        FONT_DISPLAY,
        align="center",
    )
    cv2.line(canvas, (80, 250), (DASHBOARD_W - 80, 250), (214, 208, 211), 1)
    draw_text(
        canvas,
        "PULLING OVER",
        (DASHBOARD_W // 2, 301),
        1.18,
        COLOR_TEXT,
        3,
        FONT_DISPLAY,
        align="center",
    )
    draw_text(
        canvas,
        "MICROSLEEP SAFETY PROTOCOL",
        (DASHBOARD_W // 2, 330),
        0.32,
        (228, 221, 223),
        1,
        align="center",
    )

    info_rect = (24, 354, DASHBOARD_W - 48, 174)
    rounded_rect(canvas, info_rect, (40, 25, 35), 16, -1)
    rounded_rect(canvas, info_rect, (128, 97, 113), 16, 1)
    rows = (
        ("SPEED", "0 km/h", COLOR_TEXT),
        ("STATUS", "Emergency Stop", COLOR_TEXT),
        ("MUSIC", "Paused for safety", (224, 218, 220)),
        ("AC", f"{safe_int(state.ac_temp_c)} C  /  cabin stabilised", (224, 218, 220)),
    )
    for index, (label, value, colour) in enumerate(rows):
        y = 386 + index * 34
        draw_text(canvas, label, (43, y), 0.31, (226, 207, 214), 1)
        draw_text(
            canvas,
            value,
            (DASHBOARD_W - 43, y),
            0.48,
            colour,
            1,
            FONT_DISPLAY,
            align="right",
        )

    dispatch_rect = (24, 538, DASHBOARD_W - 48, 120)
    rounded_rect(canvas, dispatch_rect, (40, 27, 39), 14, -1)
    rounded_rect(canvas, dispatch_rect, (132, 101, 117), 14, 1)
    draw_text(
        canvas,
        f"SIMULATED EMERGENCY DISPATCH  |  {state.dispatch_delay:g}s WINDOW",
        (43, 562),
        0.28,
        (232, 218, 223),
        1,
    )
    draw_badge(
        canvas,
        (DASHBOARD_W - 43, 546),
        "NO REAL CALL",
        (35, 26, 42),
        COLOR_AMBER,
        0.26,
        align="right",
    )

    if state.dispatch_phase == "countdown":
        dispatch_title = f"CALL IN {state.dispatch_remaining:04.1f}s  -  PRESS C TO CANCEL"
        dispatch_colour = COLOR_AMBER
        dispatch_detail = "Driver unresponsive  |  Press R to reset"
    elif state.dispatch_phase == "dispatched":
        dispatch_title = "SIMULATED CALL PLACED"
        dispatch_colour = COLOR_TEXT
        dispatch_detail = f"Case {state.dispatch_case_id}  |  Stored locally only"
    elif state.dispatch_phase == "cancelled":
        dispatch_title = "DISPATCH CANCELLED"
        dispatch_colour = (231, 224, 226)
        dispatch_detail = "SOS alarm stays active  |  Press R to reset"
    else:
        dispatch_title = "DISPATCH STANDBY"
        dispatch_colour = (231, 224, 226)
        dispatch_detail = "Press R to reset and resume"

    draw_text(
        canvas,
        truncate_text(dispatch_title, DASHBOARD_W - 86, 0.42, 1, FONT_DISPLAY),
        (43, 590),
        0.42,
        dispatch_colour,
        1,
        FONT_DISPLAY,
    )
    draw_text(
        canvas,
        truncate_text(
            f"LOCATION  {state.dispatch_location or 'Location not configured'}",
            DASHBOARD_W - 86,
            0.31,
        ),
        (43, 615),
        0.31,
        COLOR_TEXT if state.dispatch_location_configured else COLOR_AMBER,
        1,
    )
    draw_text(
        canvas,
        truncate_text(dispatch_detail, DASHBOARD_W - 86, 0.28),
        (43, 640),
        0.28,
        (209, 199, 204),
        1,
    )

    audio_text = "SOS ALARM SOUNDING" if state.alarm_available else "SOS AUDIO OFFLINE"
    draw_badge(
        canvas,
        (DASHBOARD_W // 2, 670),
        audio_text,
        COLOR_TEXT,
        (55, 38, 130) if state.alarm_available else (55, 34, 80),
        0.32,
        align="center",
    )
    draw_text(
        canvas,
        "Q Quit  |  SIMULATION ONLY - NO REAL CALL PLACED",
        (DASHBOARD_W // 2, 709),
        0.28,
        (225, 215, 218),
        1,
        align="center",
    )
    return canvas


def render_frame(
    frame: Optional[np.ndarray],
    state: DashboardState,
    face_box: Optional[tuple[int, int, int, int]],
    now: float,
) -> np.ndarray:
    left = render_camera_panel(frame, state, face_box, now)
    right = render_dashboard(state, now)
    canvas = np.hstack((left, right))
    cv2.line(canvas, (CAMERA_W, 0), (CAMERA_W, WINDOW_H), COLOR_BORDER, 1)
    if state.emergency:
        flash = (math.sin(now * 2.0 * math.pi * 2.4) + 1.0) * 0.5
        colour = COLOR_RED if flash > 0.35 else COLOR_RED_DARK
        cv2.rectangle(canvas, (0, 0), (WINDOW_W - 1, WINDOW_H - 1), colour, 7)
    return canvas


# ---------------------------------------------------------------------------
# Application loop
# ---------------------------------------------------------------------------


def _music_for(emotion: Optional[str], ready: bool) -> str:
    if not ready or emotion is None:
        return "Audio Standby"
    if emotion == "happy":
        return "Upbeat Music Playing"
    if emotion in RELAXING_EMOTIONS:
        return "Relaxing Music Playing"
    if emotion in {"neutral", "surprise"}:
        return "Focus Playlist Playing"
    return "Ambient Drive Playing"


def _make_emotion_crop(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
    pad = int(max(face_w, face_h) * 0.12)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    crop = frame[y1:y2, x1:x2]
    largest = max(crop.shape[:2])
    if largest > 360:
        scale = 360.0 / largest
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def _open_camera(index: int) -> cv2.VideoCapture:
    capture = None
    if os.name == "nt":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera index {index}. Close other camera apps, "
            "check Windows camera permissions, or try --camera 1."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-CoPilot - real-time webcam driver monitoring dashboard",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="eye-aspect-ratio threshold; lower values require more closed eyes (default: 0.21)",
    )
    parser.add_argument(
        "--drowsy-seconds",
        type=float,
        default=2.5,
        help="continuous eye-closure time before the alarm (default: 2.5)",
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="disable DeepFace emotion analysis to reduce CPU/RAM use",
    )
    parser.add_argument("--mute", action="store_true", help="disable the audible alarm")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not 0.10 <= args.ear_threshold <= 0.40:
        print("--ear-threshold should normally be between 0.10 and 0.40.")
    if not 0.5 <= args.drowsy_seconds <= 30.0:
        print("--drowsy-seconds should be between 0.5 and 30.0.")

    print("=" * 72)
    print("  AI-CoPilot | Real-time driver safety simulation")
    print("=" * 72)
    print("  Starting camera, Face Landmarker and the emotion worker...")
    print("  Q / Esc = quit    R = reset emergency    C = cancel simulated dispatch")
    print("=" * 72)

    try:
        capture = _open_camera(args.camera)
    except RuntimeError as exc:
        print(f"[camera] {exc}")
        return 2

    try:
        face_monitor = FaceMonitor()
    except Exception as exc:
        capture.release()
        print(f"[vision] MediaPipe Face Landmarker could not start: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3

    emotion = EmotionAnalyzer(enabled=not args.no_emotion)
    alarm = AlarmSystem(enabled=not args.mute)
    drowsiness = DrowsinessMonitor(
        ear_threshold=args.ear_threshold,
        drowsy_seconds=args.drowsy_seconds,
    )
    dispatcher = EmergencyDispatcher()

    speed = 0.0
    ac_temp = 22.0
    emergency = False
    face_found = False
    last_face_box: Optional[tuple[int, int, int, int]] = None
    last_emotion_submit = 0.0
    emotion_interval = 3.0
    fps = 0.0
    previous_time = time.monotonic()
    consecutive_camera_failures = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = capture.read()
            now = time.monotonic()
            dt = min(0.25, max(0.0, now - previous_time))
            previous_time = now
            if dt > 0:
                fps = 0.90 * fps + 0.10 * (1.0 / dt) if fps > 0 else 1.0 / dt

            if not ok or frame is None:
                consecutive_camera_failures += 1
                face_found = False
                last_face_box = None
                reading = drowsiness.update(False, None, now)
                emotion_snapshot = emotion.snapshot()
                dispatch_snapshot = dispatcher.update(now)
                state = DashboardState(
                    speed_kmh=0.0 if emergency else speed,
                    ac_temp_c=ac_temp,
                    music="Standby" if not emergency else "Paused for safety",
                    status="Emergency Stop" if emergency else "No Driver",
                    emotion=emotion_snapshot.emotion,
                    emotion_confidence=emotion_snapshot.confidence,
                    emotion_status=emotion_snapshot.status,
                    emotion_error=emotion_snapshot.error,
                    ear=None,
                    closed_seconds=reading.closed_seconds,
                    emergency=emergency,
                    face_found=False,
                    fps=fps,
                    stress_response=False,
                    alarm_available=alarm.available,
                    ear_threshold=args.ear_threshold,
                    drowsy_seconds=args.drowsy_seconds,
                    dispatch_phase=dispatch_snapshot.phase,
                    dispatch_remaining=dispatch_snapshot.remaining,
                    dispatch_delay=dispatch_snapshot.delay,
                    dispatch_location=dispatch_snapshot.location,
                    dispatch_location_configured=dispatch_snapshot.location_configured,
                    dispatch_case_id=dispatch_snapshot.case_id,
                )
                view = render_frame(None, state, None, now)
                cv2.imshow(WINDOW_NAME, view)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("c"), ord("C")):
                    dispatcher.cancel()
                if key in (ord("r"), ord("R")) and emergency:
                    emergency = False
                    alarm.stop()
                    drowsiness.reset()
                    dispatcher.reset()
                    speed = 0.0
                    consecutive_camera_failures = 0
                    print("[system] Emergency reset. Simulated vehicle resuming.")
                if consecutive_camera_failures == 90:
                    print("[camera] Video stream stopped. Check the camera connection.")
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                continue

            consecutive_camera_failures = 0
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            face_reading = face_monitor.process(frame, now)
            face_found = face_reading.found
            last_face_box = face_reading.box if face_found else None
            reading = drowsiness.update(face_found, face_reading.ear, now)

            if reading.trigger_alarm and not emergency:
                emergency = True
                speed = 0.0
                alarm.start()
                print(
                    f"[ALERT] Eyes closed for {args.drowsy_seconds:.1f} s. "
                    "Drowsiness detected; pulling over."
                )
                dispatcher.arm(now)

            if emergency:
                speed = 0.0
            else:
                speed = min(72.0, speed + 9.0 * dt)

            emotion_snapshot = emotion.snapshot()
            now = time.monotonic()
            emotion_fresh = (
                emotion_snapshot.updated_at > 0
                and now - emotion_snapshot.updated_at < 15.0
                and emotion_snapshot.emotion is not None
            )
            stress = bool(
                emotion_fresh
                and (
                    emotion_snapshot.emotion in STRESS_EMOTIONS
                    or emotion_snapshot.scores.get("angry", 0.0) >= 35.0
                    or emotion_snapshot.scores.get("fear", 0.0) >= 35.0
                )
            )
            ac_target = 20.0 if stress else 22.0
            if ac_temp < ac_target:
                ac_temp = min(ac_target, ac_temp + 1.4 * dt)
            elif ac_temp > ac_target:
                ac_temp = max(ac_target, ac_temp - 1.4 * dt)

            if not emergency and face_found and last_face_box is not None:
                if now - last_emotion_submit >= emotion_interval:
                    crop = _make_emotion_crop(frame, last_face_box)
                    if crop is not None:
                        emotion.submit(crop)
                        last_emotion_submit = now

            if emergency:
                status = "Emergency Stop"
                music = "Paused for safety"
            elif not face_found:
                status = "No Driver"
                music = _music_for(None, False)
            else:
                status = "Drowsy" if reading.drowsy else "Safe"
                music = _music_for(emotion_snapshot.emotion, emotion_fresh)

            display_emotion = emotion_snapshot.emotion if emotion_fresh else None
            display_confidence = (
                emotion_snapshot.confidence if emotion_fresh else 0.0
            )
            if emergency:
                display_emotion = emotion_snapshot.emotion

            dispatch_snapshot = dispatcher.update(now)
            state = DashboardState(
                speed_kmh=speed,
                ac_temp_c=ac_temp,
                music=music,
                status=status,
                emotion=display_emotion,
                emotion_confidence=display_confidence,
                emotion_status=emotion_snapshot.status,
                emotion_error=emotion_snapshot.error,
                ear=reading.ear,
                closed_seconds=reading.closed_seconds,
                emergency=emergency,
                face_found=face_found,
                fps=fps,
                stress_response=stress,
                alarm_available=alarm.available,
                ear_threshold=args.ear_threshold,
                drowsy_seconds=args.drowsy_seconds,
                dispatch_phase=dispatch_snapshot.phase,
                dispatch_remaining=dispatch_snapshot.remaining,
                dispatch_delay=dispatch_snapshot.delay,
                dispatch_location=dispatch_snapshot.location,
                dispatch_location_configured=dispatch_snapshot.location_configured,
                dispatch_case_id=dispatch_snapshot.case_id,
            )
            view = render_frame(frame, state, last_face_box, now)
            cv2.imshow(WINDOW_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                dispatcher.cancel()
            if key in (ord("r"), ord("R")) and emergency:
                emergency = False
                alarm.stop()
                drowsiness.reset()
                dispatcher.reset()
                speed = 0.0
                print("[system] Emergency reset. Simulated vehicle resuming.")

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        alarm.close()
        emotion.close()
        face_monitor.close()
        capture.release()
        cv2.destroyAllWindows()

    print("AI-CoPilot stopped. Drive safely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
