from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default="sqlite+aiosqlite:///./sentinellm.db")
    redis_url: str = Field(default="redis://localhost:6379/0")

    hf_model_repo: str = Field(default="distilbert-base-uncased")
    onnx_model_path: str | None = Field(default=None)

    cache_ttl_seconds: int = Field(default=86400)
    flag_threshold: float = Field(default=0.7)

    log_level: str = Field(default="INFO")

    gemini_api_key: str | None = Field(default=None)


settings = Settings()
