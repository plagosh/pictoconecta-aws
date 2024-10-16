from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
import torch

load_dotenv()

class ConfigModel(BaseSettings):
    model_checkpoint_gpt: str = "./gpt2-pretrained"
    OPENAI_API_KEY: str
    max_history: int = 10
    max_token: int = 150
    no_sample: bool = "false" in ("true", "1")
    max_length: int = 10
    min_length: int = 1
    seed: int = 0
    temperature: float = 0.6
    top_k: int = 5
    top_p: float = 0.95
    num_suggestions: int = 1
    historial_path: str = "/tmp/historial.json"
    stats_path: str = "stats_text_generation.json"
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    class Config:
        env_file_encoding = "utf-8"
        protected_namespaces = ('settings_',)
