from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding & AI Configuration
    # Options for EMBEDDING_PROVIDER: 'local' (0 cost, fast, no limits) or 'gemini' (cloud API)
    EMBEDDING_PROVIDER: str = "local"
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    GEMINI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    CHAT_MODEL: str = "models/gemini-3.6-flash"

    # Directory paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    CHROMA_DB_DIR: str = "./chroma_db"
    DATA_DIR: str = "./data"

    # RAG parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_CHUNKS: int = 4

    # Startup & Server
    AUTO_INGEST_ON_STARTUP: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a valid Gemini API key is configured."""
        key = self.GEMINI_API_KEY.strip()
        return bool(key and key != "YOUR_GEMINI_API_KEY_HERE" and len(key) >= 15)

    @property
    def resolved_chroma_dir(self) -> Path:
        """Get absolute path to chroma storage directory."""
        path = Path(self.CHROMA_DB_DIR)
        return path if path.is_absolute() else self.BASE_DIR / path

    @property
    def resolved_data_dir(self) -> Path:
        """Get absolute path to data directory."""
        path = Path(self.DATA_DIR)
        return path if path.is_absolute() else self.BASE_DIR / path


settings = Settings()
