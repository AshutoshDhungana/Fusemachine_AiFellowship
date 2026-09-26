"""Central configuration (env vars / .env)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- LLM providers (all spoken to through the OpenAI-compatible protocol) ---
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    primary_model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.5-flash-lite"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enable_vllm: bool = True

    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 1024
    request_timeout_s: float = 60.0

    # --- reliability ---
    llm_rpm: int = 8                   # client-side token bucket per provider
    retry_attempts: int = 3
    retry_base_delay_s: float = 1.5
    circuit_fail_threshold: int = 3
    circuit_cooldown_s: float = 60.0
    cache_ttl_seconds: int = 3600
    cache_max_items: int = 512
    api_rpm_per_client: int = 30

    # --- RAG ---
    papers_dir: Path = ROOT / "data" / "papers"
    catalog_path: Path = ROOT / "data" / "papers.yaml"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "papers"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chunk_words: int = 300
    chunk_overlap_words: int = 60
    top_k: int = 5
    max_top_k: int = 8                 # hard cap (context engineering)
    chunk_char_cap: int = 900          # per-chunk truncation in tool output

    # --- W14 router (ONNX) ---
    router_dir: Path = ROOT / "artifacts" / "router_onnx"
    router_max_batch: int = 32
    router_max_wait_ms: float = 5.0

    # --- Agent (W16) ---
    agent_prompt: str = "agent_v1"
    agent_max_steps: int = 8
    agent_max_tokens: int = 60_000
    agent_verify: bool = True
    tool_timeout_s: float = 20.0
    fail_inject: str = ""              # "", "unavailable", "malformed", "timeout"


@lru_cache
def get_settings() -> Settings:
    return Settings()
