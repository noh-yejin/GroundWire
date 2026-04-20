from __future__ import annotations

import logging
import re
import time

from pydantic import BaseModel, Field

from app.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in tests
    OpenAI = None

logger = logging.getLogger(__name__)

FALLBACK_TRANSLATIONS = {
    "live updates": "실시간 업데이트",
    "stock futures": "주가지수 선물",
    "dow futures": "다우 선물",
    "nasdaq": "나스닥",
    "s&p 500": "S&P500",
    "benchmark lending rates": "기준 대출금리",
    "lending rates": "대출금리",
    "rate cut": "금리 인하",
    "rate hike": "금리 인상",
    "inflation": "인플레이션",
    "economic growth": "경제 성장",
    "middle east": "중동",
    "risks loom": "리스크가 커지는 가운데",
    "markets wrap": "시장 종합",
    "oil prices": "유가",
    "war tensions": "전쟁 긴장",
    "ceasefire": "휴전",
}


class UILocalizationSchema(BaseModel):
    text: str = Field(description="Natural Korean UI text")


class UIDisplayLocalizer:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=0) if (
            settings.openai_api_key and OpenAI is not None
        ) else None
        self._cache: dict[tuple[str, str, int], str] = {}
        self._remote_disabled_until: float = 0.0

    def localize_label(self, text: str, *, allow_remote: bool = True) -> str:
        return self._localize(text, mode="label", max_chars=28, allow_remote=allow_remote)

    def localize_summary(self, text: str, *, allow_remote: bool = True) -> str:
        return self._localize(text, mode="summary", max_chars=140, allow_remote=allow_remote)

    def localize_detail(self, text: str, *, allow_remote: bool = True) -> str:
        return self._localize(text, mode="detail", max_chars=4000, allow_remote=allow_remote)

    def localize_point(self, text: str, max_chars: int = 30, *, allow_remote: bool = True) -> str:
        return self._localize(text, mode="point", max_chars=max_chars, allow_remote=allow_remote)

    def _localize(self, text: str, *, mode: str, max_chars: int, allow_remote: bool) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""
        cache_key = (mode, cleaned, max_chars)
        if cache_key in self._cache:
            return self._cache[cache_key]

        needs_llm = bool(re.search(r"[A-Za-z]", cleaned)) or (mode == "point" and len(cleaned) > max_chars)
        remote_available = allow_remote and self.client is not None and time.time() >= self._remote_disabled_until
        if not needs_llm or not remote_available:
            result = self._fallback(cleaned, mode=mode, max_chars=max_chars)
            self._cache[cache_key] = result
            return result

        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You convert financial/news UI text into natural Korean for display only. "
                            "Do not add facts. Keep meaning faithful. "
                            "For labels, keep them very short. "
                            "For points, compress the whole sentence into one concise Korean phrase instead of just truncating the first sentence. "
                            "Return only one Korean string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"mode={mode}\n"
                            f"max_chars={max_chars}\n"
                            f"text={cleaned}\n"
                            "규칙:\n"
                            "- 영어는 자연스러운 한국어로 번역\n"
                            "- 고유명사는 필요할 때 유지\n"
                            "- point 모드에서는 문장 전체 의미를 유지한 짧은 한 문장으로 압축\n"
                            "- summary 모드는 핵심만 자연스럽게 한국어로 정리\n"
                        ),
                    },
                ],
                text_format=UILocalizationSchema,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed and parsed.text:
                result = self._fallback(parsed.text, mode=mode, max_chars=max_chars)
                self._cache[cache_key] = result
                return result
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("UI localization failed: %s: %s", type(exc).__name__, exc)
            self._remote_disabled_until = time.time() + 60

        result = self._fallback(cleaned, mode=mode, max_chars=max_chars)
        self._cache[cache_key] = result
        return result

    def _fallback(self, text: str, *, mode: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = re.sub(r"\s*-\s*(Reuters|Bloomberg|CNBC|WSJ|연합뉴스|한국경제|매일경제)$", "", cleaned, flags=re.IGNORECASE)
        translated = cleaned
        for source, target in FALLBACK_TRANSLATIONS.items():
            translated = re.sub(re.escape(source), target, translated, flags=re.IGNORECASE)
        translated = re.sub(r"\s+", " ", translated).strip(" .")
        if mode == "point":
            compact = re.sub(r"\([^)]*\)", "", translated)
            compact = re.sub(r"\[[^\]]*\]", "", compact)
            compact = re.sub(r"^[\-\u2022]+\s*", "", compact)
            compact = re.sub(r":\s*실시간 업데이트$", "", compact)
            compact = compact.strip()
            if len(compact) <= max_chars:
                return compact
            return f"{compact[:max_chars].rstrip()}..."
        if len(translated) <= max_chars:
            return translated
        clipped = translated[:max_chars].rstrip()
        clipped = re.sub(r"[,:;]\s*$", "", clipped)
        return f"{clipped}..."


ui_localizer = UIDisplayLocalizer()
