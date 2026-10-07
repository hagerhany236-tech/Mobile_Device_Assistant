"""LLM extraction + Pydantic validation + retry loop."""
import json
import os
from functools import lru_cache
from typing import Dict, List

from pydantic import ValidationError

from app.prompts.templates import SYSTEM_PROMPT, build_retry_message, build_user_message
from app.schema.device import Device

MAX_RETRIES = 2 


class ExtractionError(Exception):
    """Raised when the LLM cannot produce a valid Device after all retries."""


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    return OpenAI(api_key=api_key)


def _call_llm(messages: List[Dict[str, str]]) -> str:
    response = _get_client().chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content or ""


def extract_device(text: str) -> Device:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(text)},
    ]
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        raw = _call_llm(messages)
        try:
            return Device.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
             
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": build_retry_message(last_error)})
    raise ExtractionError(
        f"Could not obtain valid output after {MAX_RETRIES} retries. Last error: {last_error}"
    )
