"""One small API that recognizes  through the Iriun camera."""

import hashlib
import threading
import time
from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import cv2
import face_recognition
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from camera import open_camera
from face_service import __version__
from face_service.config import Settings, get_settings
from face_service.services.recognition import FaceImageError, FaceRecognitionService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNSCALE = 0.5
PROCESS_EVERY = 2
ENCODER_MODEL = "large"
PREVIEW_INTERVAL_SECONDS = 0.12
MAX_JPEG_BYTES = 1_500_000

recognizer = FaceRecognitionService(encoder_model=ENCODER_MODEL, enrol_jitters=10)
recognition_lock = threading.Lock()
preview_lock = threading.Lock()
latest_preview_jpeg: bytes | None = None


class RecognitionStatus(StrEnum):
    MATCHED = auto()
    UNKNOWN = auto()
    NO_FACE = auto()
    CAMERA_ERROR = auto()
    PROCESSING_ERROR = auto()


class RegistrationView(StrEnum):
    FRONT = auto()
    SIDE = auto()


class RecognitionResponse(BaseModel):
    """The complete answer returned to the calling software."""

    model_config = ConfigDict(populate_by_name=True)

    approved: bool
    status: RecognitionStatus
    person: str | None = None
    distance: float | None = Field(default=None, ge=0)
    camera_index: int = Field(alias="cameraIndex")
    frames_scanned: int = Field(alias="framesScanned", ge=0)
    message: str


class RegistrationResponse(BaseModel):
    """The result of storing one reference view for a person."""

    model_config = ConfigDict(populate_by_name=True)

    person: str
    view: RegistrationView
    stored_as: str = Field(alias="storedAs")
    captured_views: list[RegistrationView] = Field(alias="capturedViews")
    complete: bool
    message: str


class RegistrationStatusResponse(BaseModel):
    """Whether the session UI should register or verify this person."""

    model_config = ConfigDict(populate_by_name=True)

    person: str
    registered: bool
    captured_views: list[RegistrationView] = Field(alias="capturedViews")
    reference_count: int = Field(alias="referenceCount", ge=0)


def publish_preview(frame: Any | None) -> None:
    """Keep only the latest camera frame in RAM; nothing is written to disk."""

    global latest_preview_jpeg
    encoded: bytes | None = None
    if frame is not None:
        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 78],
        )
        if ok:
            encoded = jpeg.tobytes()

    with preview_lock:
        latest_preview_jpeg = encoded


def read_preview() -> bytes | None:
    with preview_lock:
        return latest_preview_jpeg


def reference_fingerprint(image_paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()

    for image_path in image_paths:
        file_info = image_path.stat()
        description = f"{image_path.name}:{file_info.st_size}:{file_info.st_mtime_ns}"
        digest.update(description.encode("utf-8"))
    return digest.hexdigest()


def validate_person_name(person_name: str) -> str:
    """Allow a readable folder name without permitting path traversal."""

    cleaned = person_name.strip()
    if not cleaned or len(cleaned) > 64:
        raise ValueError("Person name must contain between 1 and 64 characters")
    if not any(character.isalnum() for character in cleaned):
        raise ValueError("Person name must contain a letter or number")
    if any(
        not (character.isalnum() or character in {" ", "_", "-"})
        for character in cleaned
    ):
        raise ValueError(
            "Person name may contain only letters, numbers, spaces, _ or -"
        )
    return cleaned


def registration_status(person_name: str) -> RegistrationStatusResponse:
    person_directory = PROJECT_ROOT / "known_faces" / person_name
    if not person_directory.is_dir():
        return RegistrationStatusResponse(
            person=person_name,
            registered=False,
            capturedViews=[],
            referenceCount=0,
        )

    image_paths = [
        image_path
        for image_path in person_directory.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    captured_views = [
        view
        for view in RegistrationView
        if any(
            image_path.name.startswith(f"{view.value}-") for image_path in image_paths
        )
    ]
    labelled_names = tuple(f"{view.value}-" for view in RegistrationView)
    has_legacy_references = any(
        not image_path.name.startswith(labelled_names) for image_path in image_paths
    )
    registered = has_legacy_references or len(captured_views) == len(RegistrationView)
    return RegistrationStatusResponse(
        person=person_name,
        registered=registered,
        capturedViews=captured_views,
        referenceCount=len(image_paths),
    )


@lru_cache
def load_reference_encodings(person_name: str) -> tuple[Any, ...]:
    """Load and encode known_faces/<person_name> once per server process."""

    person_directory = PROJECT_ROOT / "known_faces" / person_name
    if not person_directory.is_dir():
        raise RuntimeError(f"Reference folder does not exist: {person_directory}")

    image_paths = tuple(
        image_path
        for image_path in sorted(person_directory.iterdir())
        if image_path.is_file()
        and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise RuntimeError(f"No references photos found for {person_name}")

    fingerprint = reference_fingerprint(image_paths)
    cache_path = PROJECT_ROOT / f".{person_name}_encodings.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                saved_fingerprint = str(cache["fingerprint"].item())

                if saved_fingerprint == fingerprint:
                    print(f"Loaded cached encodings for {person_name}")
                    return tuple(cache["encodings"])

                print("Reference photos changed; rebuilding cache")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Could not read encoding cache: {exc}")

    encodings: list[Any] = []
    for image_path in image_paths:
        try:
            encoding = recognizer.encode_one(
                image_path.read_bytes(),
                enrolment=True,
            )
        except FaceImageError as exc:
            print(f"Skipped {image_path.name}: {exc}")
            continue
        encodings.append(encoding)
        print(f"Loaded reference: {image_path.name} -> {person_name}")

    if not encodings:
        raise RuntimeError(f"No usable reference faces found for {person_name}")

    np.savez_compressed(
        cache_path,
        fingerprint=fingerprint,
        encodings=np.stack(encodings),
    )
    print(f"Saved encoding cache for {person_name}: {cache_path.name}")

    return tuple(encodings)


def result(
    settings: Settings,
    *,
    status: RecognitionStatus,
    frames_scanned: int,
    message: str,
    distance: float | None = None,
) -> RecognitionResponse:
    """Build one consistent API response."""

    approved = status == RecognitionStatus.MATCHED
    return RecognitionResponse(
        approved=approved,
        status=status,
        person=settings.person_name if approved else None,
        distance=distance,
        cameraIndex=settings.camera_index,
        framesScanned=frames_scanned,
        message=message,
    )


def register_face_bytes(
    image_bytes: bytes,
    person_name: str,
    view: RegistrationView,
) -> RegistrationResponse:
    """Validate and store one front/side reference JPEG."""

    recognizer.encode_one(image_bytes, enrolment=True)

    person_directory = PROJECT_ROOT / "known_faces" / person_name
    person_directory.mkdir(parents=True, exist_ok=True)
    filename = f"{view.value}-{uuid4().hex[:12]}.jpg"
    (person_directory / filename).write_bytes(image_bytes)

    # A running process may already have cached this person's old references.
    load_reference_encodings.cache_clear()

    captured_views = [
        candidate
        for candidate in RegistrationView
        if any(person_directory.glob(f"{candidate.value}-*.jpg"))
    ]
    complete = len(captured_views) == len(RegistrationView)
    return RegistrationResponse(
        person=person_name,
        view=view,
        storedAs=filename,
        capturedViews=captured_views,
        complete=complete,
        message=(
            f"{person_name} now has front and side reference photos"
            if complete
            else f"Stored {view.value} view; the other view is still required"
        ),
    )


def scan_camera_for_person(settings: Settings) -> RecognitionResponse:
    """Open Iriun, scan briefly for , and always release the camera."""

    publish_preview(None)

    try:
        known_encodings = load_reference_encodings(settings.person_name)
    except Exception as exc:
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=0,
            message=str(exc),
        )

    camera, _backend = open_camera(settings.camera_index)
    if camera is None:
        return result(
            settings,
            status=RecognitionStatus.CAMERA_ERROR,
            frames_scanned=0,
            message=f"Could not open camera {settings.camera_index}",
        )

    deadline = time.monotonic() + settings.scan_timeout_seconds
    frames_scanned = 0
    saw_face = False
    nearest_distance: float | None = None
    scale = round(1 / DOWNSCALE)
    next_preview_at = 0.0

    try:
        while time.monotonic() < deadline:
            captured, frame = camera.read()
            if not captured:
                return result(
                    settings,
                    status=RecognitionStatus.CAMERA_ERROR,
                    frames_scanned=frames_scanned,
                    message="The camera opened but did not return a frame",
                )

            frames_scanned += 1
            now = time.monotonic()
            if now >= next_preview_at:
                publish_preview(frame)
                next_preview_at = now + PREVIEW_INTERVAL_SECONDS

            if frames_scanned % PROCESS_EVERY != 0:
                continue

            rgb_full = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rgb_small = np.ascontiguousarray(
                cv2.resize(rgb_full, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            )
            small_boxes = face_recognition.face_locations(rgb_small)
            boxes = [
                (top * scale, right * scale, bottom * scale, left * scale)
                for top, right, bottom, left in small_boxes
            ]
            probe_encodings = face_recognition.face_encodings(
                rgb_full,
                boxes,
                model=ENCODER_MODEL,
            )

            if probe_encodings:
                saw_face = True

            for probe_encoding in probe_encodings:
                nearest = recognizer.nearest(probe_encoding, known_encodings)
                if nearest is None:
                    continue

                _best_index, distance = nearest
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance

                if distance <= settings.match_tolerance:
                    return result(
                        settings,
                        status=RecognitionStatus.MATCHED,
                        frames_scanned=frames_scanned,
                        distance=distance,
                        message=f"{settings.person_name} recognized",
                    )
    finally:
        camera.release()

    if saw_face:
        return result(
            settings,
            status=RecognitionStatus.UNKNOWN,
            frames_scanned=frames_scanned,
            distance=nearest_distance,
            message=f"A face was found, but it was not {settings.person_name}",
        )

    return result(
        settings,
        status=RecognitionStatus.NO_FACE,
        frames_scanned=frames_scanned,
        message="No face was found before the scan timed out",
    )


def recognize_frame_bytes(
    image_bytes: bytes,
    settings: Settings,
) -> RecognitionResponse:
    """Recognize one JPEG captured by the remote shooter browser."""

    try:
        known_encodings = load_reference_encodings(settings.person_name)
        probe_encodings = recognizer.encode_faces(image_bytes)
    except FaceImageError as exc:
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=1,
            message=str(exc),
        )
    except Exception as exc:
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=1,
            message=str(exc),
        )

    if not probe_encodings:
        return result(
            settings,
            status=RecognitionStatus.NO_FACE,
            frames_scanned=1,
            message="No face was found in the shooter camera frame",
        )

    if len(probe_encodings) != 1:
        return result(
            settings,
            status=RecognitionStatus.UNKNOWN,
            frames_scanned=1,
            message="More than one face was found in the shooter camera frame",
        )

    nearest = recognizer.nearest(probe_encodings[0], known_encodings)
    if nearest is None:
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=1,
            message="No reference encodings are available",
        )

    _best_index, distance = nearest
    if distance <= settings.match_tolerance:
        return result(
            settings,
            status=RecognitionStatus.MATCHED,
            frames_scanned=1,
            distance=distance,
            message=f"{settings.person_name} recognized",
        )

    return result(
        settings,
        status=RecognitionStatus.UNKNOWN,
        frames_scanned=1,
        distance=distance,
        message=f"A face was found, but it was not {settings.person_name}",
    )


app = FastAPI(
    title=" Face Recognition API",
    version=__version__,
    description="A local learning API that approves only .",
)


@app.get("/health", tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    """Confirm the API process is running; this does not open the camera."""

    return {
        "status": "ok",
        "service": settings.service_name,
        "version": __version__,
        "person": settings.person_name,
        "cameraIndex": settings.camera_index,
    }


@app.post("/recognize", response_model=RecognitionResponse, tags=["recognition"])
async def recognize(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RecognitionResponse:
    """Open the camera, look for , close it, and return the answer."""

    if not recognition_lock.acquire(blocking=False):
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=0,
            message="A recognition scan is already running",
        )

    try:
        return await run_in_threadpool(scan_camera_for_person, settings)
    finally:
        recognition_lock.release()


@app.post(
    "/recognize-frame",
    response_model=RecognitionResponse,
    tags=["recognition"],
)
async def recognize_frame(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    person_name: Annotated[str | None, Query(alias="personName")] = None,
) -> RecognitionResponse:
    """Recognize one JPEG held in memory; never save the frame to disk."""

    if person_name is not None:
        try:
            safe_person_name = validate_person_name(person_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        settings = settings.model_copy(update={"person_name": safe_person_name})

    if request.headers.get("content-type", "").split(";", 1)[0] != "image/jpeg":
        raise HTTPException(status_code=415, detail="A JPEG camera frame is required")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Camera frame is empty")
    if len(image_bytes) > MAX_JPEG_BYTES:
        raise HTTPException(status_code=413, detail="Camera frame is too large")

    if not recognition_lock.acquire(blocking=False):
        return result(
            settings,
            status=RecognitionStatus.PROCESSING_ERROR,
            frames_scanned=0,
            message="A recognition scan is already running",
        )

    try:
        return await run_in_threadpool(recognize_frame_bytes, image_bytes, settings)
    finally:
        recognition_lock.release()


@app.get(
    "/face-registration/{person_name}",
    response_model=RegistrationStatusResponse,
    tags=["registration"],
)
def get_face_registration(person_name: str) -> RegistrationStatusResponse:
    """Tell the session UI whether to show Register Face or Verify Face."""

    try:
        safe_person_name = validate_person_name(person_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return registration_status(safe_person_name)


@app.post(
    "/register-face/{person_name}/{view}",
    response_model=RegistrationResponse,
    tags=["registration"],
)
async def register_face(
    person_name: str,
    view: RegistrationView,
    request: Request,
) -> RegistrationResponse:
    """Register one front or side JPEG for the admin-provided person name."""

    try:
        safe_person_name = validate_person_name(person_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if request.headers.get("content-type", "").split(";", 1)[0] != "image/jpeg":
        raise HTTPException(
            status_code=415, detail="A JPEG reference photo is required"
        )

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Reference photo is empty")
    if len(image_bytes) > MAX_JPEG_BYTES:
        raise HTTPException(status_code=413, detail="Reference photo is too large")

    if not recognition_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Recognition is currently running")

    try:
        try:
            return await run_in_threadpool(
                register_face_bytes,
                image_bytes,
                safe_person_name,
                view,
            )
        except FaceImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        recognition_lock.release()


@app.get("/preview", tags=["recognition"])
def preview() -> Response:
    """Return the latest frame used by recognition without opening the camera."""

    jpeg = read_preview()
    headers = {"Cache-Control": "no-store, max-age=0"}
    if jpeg is None:
        return Response(status_code=204, headers=headers)
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)
