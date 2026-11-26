"""LLM provider abstraction with ordered fallback, e.g. Claude -> Gemini -> Ollama.

Providers are imported lazily, so you only need the SDK + key for those you enable.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str: ...


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: int):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def generate(self, system, prompt, max_tokens, temperature):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature, system=system,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError("Claude returned an empty response")
        return text


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: int):
        from google import genai
        from google.genai import types

        self.model, self._types = model, types
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000))

    def generate(self, system, prompt, max_tokens, temperature):
        cfg = self._types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                max_output_tokens=max_tokens)
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        text = (resp.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty/blocked response")
        return text


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: int):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def generate(self, system, prompt, max_tokens, temperature):
        r = httpx.post(f"{self.url}/api/chat", timeout=self.timeout, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"temperature": temperature, "num_predict": max_tokens}})
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        if not text:
            raise LLMError("Ollama returned an empty response")
        return text


class LLMRouter:
    """Tries providers in order; the first success wins."""

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    def generate(self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> LLMResult:
        if not self.providers:
            raise LLMError("No LLM providers configured (check API keys / ANSWER_PROVIDERS)")
        errors = []
        for p in self.providers:
            try:
                text = p.generate(system, prompt, max_tokens, temperature)
                if errors:
                    log.warning("Answered by fallback provider %s after: %s", p.name, "; ".join(errors))
                return LLMResult(text, p.name)
            except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
                log.warning("LLM provider %s failed: %s", p.name, exc)
                errors.append(f"{p.name}: {exc}")
        raise LLMError("All LLM providers failed - " + "; ".join(errors))


def build_router(names_csv: str, s) -> LLMRouter:
    providers: list[LLMProvider] = []
    for name in [n.strip().lower() for n in names_csv.split(",") if n.strip()]:
        try:
            if name == "claude":
                if not s.anthropic_api_key:
                    log.warning("Skipping claude: ANTHROPIC_API_KEY not set")
                    continue
                providers.append(ClaudeProvider(s.anthropic_api_key, s.claude_model, s.llm_timeout_s))
            elif name == "gemini":
                if not s.gemini_api_key:
                    log.warning("Skipping gemini: GEMINI_API_KEY not set")
                    continue
                providers.append(GeminiProvider(s.gemini_api_key, s.gemini_model, s.llm_timeout_s))
            elif name == "ollama":
                providers.append(OllamaProvider(s.ollama_url, s.ollama_model, s.llm_timeout_s))
            else:
                log.warning("Unknown LLM provider '%s' ignored", name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not initialise provider %s: %s", name, exc)
    return LLMRouter(providers)


def configured_providers(s) -> list[str]:
    out = []
    for n in [x.strip().lower() for x in s.answer_providers.split(",") if x.strip()]:
        if (n == "claude" and s.anthropic_api_key) or (n == "gemini" and s.gemini_api_key) or n == "ollama":
            out.append(n)
    return out
