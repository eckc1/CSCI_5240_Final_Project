import json
from typing import Optional
from openai import OpenAI
from src.config import settings

def has_llm() -> bool:
    return bool(settings.openai_api_key)

def chat_json(system_prompt: str, user_prompt: str) -> Optional[dict]:
    if not has_llm():
        return None
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return json.loads(content) if content else None
    except Exception:
        return None
