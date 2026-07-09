from typing import Optional
DEFAULT_GOOGLE_MODEL = 'gemini-2.0-flash'
GOOGLE_MODEL_ALIASES = {'gemini-2.5-pro-preview-06-05': 'gemini-2.5-flash', 'gemini-2.5-flash-preview-05-20': 'gemini-2.5-flash', 'gemini-pro': DEFAULT_GOOGLE_MODEL, 'gemini-pro-vision': DEFAULT_GOOGLE_MODEL, 'gemini-1.5-pro-latest': DEFAULT_GOOGLE_MODEL, 'gemini-1.5-flash-latest': DEFAULT_GOOGLE_MODEL}

def normalize_google_model(model_name: Optional[str], default: str=DEFAULT_GOOGLE_MODEL) -> str:
    raw = str(model_name or '').strip()
    if not raw:
        return default
    normalized = raw.lower()
    if normalized.startswith('models/'):
        normalized = normalized.split('/', 1)[1]
    mapped = GOOGLE_MODEL_ALIASES.get(normalized, normalized)
    if mapped.startswith('models/'):
        mapped = mapped.split('/', 1)[1]
    return mapped or default
