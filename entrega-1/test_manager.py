import asyncio

import pytest

from llm.exceptions import LLMProviderError, LLMRateLimitError  # noqa: E402
from llm.manager import AsyncLLMManager  # noqa: E402
from llm.mock_clients import MockAnthropic, MockOpenAI  # noqa: E402
from schemas import LLMConfig, LLMProvider  # noqa: E402


def make_manager(openai_client):
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1",
        max_retries=1,
        retry_delay=0.01,
    )
    return AsyncLLMManager(
        config,
        clients={
            LLMProvider.OPENAI: openai_client,
            LLMProvider.ANTHROPIC: MockAnthropic(text="Plan B", delay=0),
        },
    )


async def collect(manager, prompt):
    return "".join([f async for f in manager.generate(prompt)])


def test_modo_normal():
    manager = make_manager(MockOpenAI(text="Hola mundo", delay=0))
    assert asyncio.run(manager.generate_text("hola")) == "Hola mundo"


def test_streaming():
    manager = make_manager(MockOpenAI(text="Hola mundo", delay=0))
    assert asyncio.run(collect(manager, "hola")) == "Hola mundo"


def test_failover_modo_normal_y_streaming():
    manager = make_manager(MockOpenAI(fail=True))
    assert asyncio.run(manager.generate_text("hola")) == "Plan B"

    manager = make_manager(MockOpenAI(fail=True))
    assert asyncio.run(collect(manager, "hola")) == "Plan B"


def test_error_controlado_si_todo_falla():
    manager = make_manager(MockOpenAI(fail=True))
    del manager.clients[LLMProvider.ANTHROPIC]

    with pytest.raises(LLMRateLimitError):
        asyncio.run(manager.generate_text("hola"))


def test_prompt_vacio():
    manager = make_manager(MockOpenAI(text="Hola mundo", delay=0))

    with pytest.raises(ValueError):
        asyncio.run(manager.generate_text("   "))


def test_sin_proveedores_disponibles():
    manager = make_manager(MockOpenAI(text="Hola mundo", delay=0))
    manager.clients.clear()

    with pytest.raises(LLMProviderError):
        asyncio.run(manager.generate_text("hola"))
