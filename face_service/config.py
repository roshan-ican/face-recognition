"""Environment-backed settings for the face-recognition service."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration read from environment variables or a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FACE_",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = " Face Recognition API"
    person_name: str = ""
    camera_index: int = Field(default=0, ge=0)
    scan_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    match_tolerance: float = Field(default=0.6, gt=0, lt=1)


@lru_cache
def get_settings() -> Settings:
    """Create settings once per process; tests can clear this cache."""

    return Settings()
