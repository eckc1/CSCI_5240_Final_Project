from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseModel):
    usda_api_key: str | None = os.getenv("USDA_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    instacart_dir: str = str(ROOT / "data" / "instacart")
    local_price_path: str = str(ROOT / "data" / "price_defaults.csv")

settings = Settings()
