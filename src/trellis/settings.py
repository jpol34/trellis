from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # RunPod (used by pod/ scripts to spin up/down the fine-tuning GPU backend)
    runpod_api_key: str = ""
    runpod_pod_id: str = ""
    pod_max_hours: int = 4
    runpod_image: str = ""
    runpod_gpu_type_id: str = "NVIDIA GeForce RTX 4090"
    runpod_disk_gb: int = 20
    runpod_registry_id: str = ""

    # LLM providers used for synthetic transcript generation and as reference benchmarks
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    typesafe_api_key: str = ""
    generation_claude_model: str = ""
    generation_concurrency: int = 30
    reference_gpt_model: str = "gpt-5.1"
    jev_base_url: str = ""

    # Fine-tuning / inference device
    trellis_device: str = "cpu"
    trellis_cpu_threads: int = 4

    # Local data locations
    checkpoint_dir: str = ""
    training_corpus_dir: str = ""
    eval_set_dir: str = ""


settings = Settings()
