from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str
    openai_webhook_secret: str
    admin_token: str

    restaurant_name: str = "Example Restaurant"

    live_model: str = "gpt-live-1"
    live_voice: str = "marin"
    backend_model: str = "gpt-6-luna"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512

    db_path: str = "./data/knowledge.sqlite3"
    auto_bootstrap_kb: bool = True
    bundled_kb_path: str = "./data/McDonalds_About_Our_Food_Summary.txt"
    bundled_kb_source: str = "mcdonalds-about-our-food-2026-10-06"

    log_level: str = "INFO"
    log_transcripts: bool = False
    greet_on_connect: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
