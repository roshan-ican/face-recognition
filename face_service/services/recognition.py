"""Reusable face encoding and nearest-neighbour matching.

The HTTP layer will translate these low-level outcomes into API status models.
Keeping this module unaware of FastAPI also makes it usable by the existing CLI.
"""

from collections.abc import Sequence
from typing import Any


class FaceImageError(ValueError):
    """Base class for an image that cannot provide one usable face."""


class FaceRecognitionService:
    """Encode images and compare encodings using the existing dlib stack."""

    def __init__(self, *, encoder_model: str = "large", enrol_jitters: int = 10):
        self.encoder_model = encoder_model
        self.enrol_jitters = enrol_jitters

    @staticmethod
    def _libraries() -> tuple[Any, Any, Any]:
        """Import heavy native libraries only when recognition is requested."""

        import cv2
        import face_recognition
        import numpy as np

        return cv2, face_recognition, np

    def decode_rgb(self, image_bytes: bytes) -> Any:
        """Decode JPEG/PNG bytes into the contiguous RGB array dlib expects."""

        cv2, _face_recognition, np = self._libraries()
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FaceImageError("The uploaded file is not a readable image")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def encode_faces(self, image_bytes: bytes, *, enrolment: bool = False) -> list[Any]:
        """Return every face encoding found in an image."""

        _cv2, face_recognition, _np = self._libraries()
        rgb = self.decode_rgb(image_bytes)
        # Registration happens only twice per shooter, so spend a little more
        # CPU here to find smaller/softer camera faces. Live verification keeps
        # the normal single upsample for speed.
        boxes = face_recognition.face_locations(
            rgb,
            number_of_times_to_upsample=2 if enrolment else 1,
        )
        jitters = self.enrol_jitters if enrolment else 1
        return list(
            face_recognition.face_encodings(
                rgb,
                boxes,
                num_jitters=jitters,
                model=self.encoder_model,
            )
        )

    def encode_one(self, image_bytes: bytes, *, enrolment: bool = False) -> Any:
        """Require exactly one detected face and return its encoding."""

        encodings = self.encode_faces(image_bytes, enrolment=enrolment)
        if not encodings:
            raise FaceImageError("No face was detected in the image")
        if len(encodings) > 1:
            raise FaceImageError(
                f"Expected one face but detected {len(encodings)}"
            )
        return encodings[0]

    def nearest(
        self,
        probe_encoding: Any,
        known_encodings: Sequence[Any],
    ) -> tuple[int, float] | None:
        """Return ``(index, distance)`` for the closest known encoding."""

        if not known_encodings:
            return None

        _cv2, face_recognition, np = self._libraries()
        distances = face_recognition.face_distance(known_encodings, probe_encoding)
        best_index = int(np.argmin(distances))
        return best_index, float(distances[best_index])


recognizer = FaceRecognitionService()


def get_recognizer() -> FaceRecognitionService:
    """Return the process-wide recognizer for FastAPI dependency injection."""

    return recognizer
