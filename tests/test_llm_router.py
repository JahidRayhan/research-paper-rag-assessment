import pytest

from src.services.llm import LLMError, LLMProvider, LLMRouter


class Ok(LLMProvider):
    def __init__(self, name): self.name = name
    def generate(self, system, prompt, max_tokens, temperature): return f"hello from {self.name}"


class Boom(LLMProvider):
    def __init__(self, name): self.name = name
    def generate(self, system, prompt, max_tokens, temperature): raise RuntimeError("429 rate limited")


def test_primary_used_first():
    assert LLMRouter([Ok("claude"), Ok("gemini")]).generate("s", "p").provider == "claude"


def test_falls_back_to_gemini():
    r = LLMRouter([Boom("claude"), Ok("gemini")]).generate("s", "p")
    assert (r.provider, r.text) == ("gemini", "hello from gemini")


def test_all_fail_raises():
    with pytest.raises(LLMError, match="claude.*gemini"):
        LLMRouter([Boom("claude"), Boom("gemini")]).generate("s", "p")


def test_no_providers():
    with pytest.raises(LLMError, match="No LLM providers"):
        LLMRouter([]).generate("s", "p")
