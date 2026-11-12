"""
Clientes falsos de OpenAI y Anthropic para probar sin API keys.

Imitan la interfaz mínima que usa AsyncLLMManager: respuesta completa,
streaming por fragmentos y, opcionalmente, un error de rate limit.
"""

import asyncio
from types import SimpleNamespace

import anthropic
import openai


def _rate_limit_response():
    """Respuesta HTTP 429 mínima para construir el error del SDK."""
    return SimpleNamespace(request=None, status_code=429, headers={})


def _fragments(text):
    """Divide el texto en palabras conservando los espacios entre ellas."""
    words = text.split(" ")
    return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]


class MockOpenAI:
    """Cliente falso de OpenAI (chat.completions)."""

    def __init__(self, text="Python es un lenguaje de programación.", fail=False, delay=0.05):
        self.text = text
        self.fail = fail
        self.delay = delay
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        if self.fail:
            raise openai.RateLimitError(
                "Rate limit simulado (mock).",
                response=_rate_limit_response(),
                body=None,
            )

        if kwargs.get("stream"):
            return self._stream()

        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def _stream(self):
        for fragment in _fragments(self.text):
            await asyncio.sleep(self.delay)
            delta = SimpleNamespace(content=fragment)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    async def close(self):
        pass


class MockAnthropic:
    """Cliente falso de Anthropic (messages)."""

    def __init__(self, text="Python es un lenguaje de programación.", fail=False, delay=0.05):
        self.text = text
        self.fail = fail
        self.delay = delay
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        if self.fail:
            raise anthropic.RateLimitError(
                "Rate limit simulado (mock).",
                response=_rate_limit_response(),
                body=None,
            )

        if kwargs.get("stream"):
            return self._stream()

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)]
        )

    async def _stream(self):
        for fragment in _fragments(self.text):
            await asyncio.sleep(self.delay)
            delta = SimpleNamespace(type="text_delta", text=fragment)
            yield SimpleNamespace(type="content_block_delta", delta=delta)

    async def close(self):
        pass
