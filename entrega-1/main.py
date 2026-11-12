import asyncio
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from llm.exceptions import LLMError
from llm.manager import AsyncLLMManager
from llm.mock_clients import MockAnthropic, MockOpenAI
from schemas import LLMConfig, LLMProvider

load_dotenv()

# Modelos por defecto para cada proveedor.
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4.1",
    LLMProvider.ANTHROPIC: "claude-sonnet-4-6",
}

PROMPT = "Explicame brevemente qué es Python en dos frases."


def build_config() -> LLMConfig:
    """Arma la configuración leyendo el proveedor desde LLM_PROVIDER."""
    provider = LLMProvider(os.getenv("LLM_PROVIDER", "openai").lower())

    return LLMConfig(
        provider=provider,
        model=DEFAULT_MODELS[provider],
        max_tokens=300,
    )


def build_manager(config: LLMConfig) -> AsyncLLMManager:
    """
    Crea el administrador. Con LLM_MOCK activo usa clientes falsos
    (sin API keys): LLM_MOCK=1 responde bien, LLM_MOCK=fail simula que el
    proveedor principal falla por rate limit y se activa el failover.
    """
    mock = os.getenv("LLM_MOCK", "").lower()

    if mock in ("", "0", "false", "no"):
        return AsyncLLMManager(config)

    primary_fails = mock == "fail"
    backup_text = "Respuesta del proveedor alternativo (failover)."

    def mock_for(provider, client_class):
        """Cliente falso: falla si es el principal y se pidió el modo fail."""
        is_primary = provider == config.provider
        return client_class(
            fail=primary_fails and is_primary,
            **({"text": backup_text} if primary_fails and not is_primary else {}),
        )

    clients = {
        LLMProvider.OPENAI: mock_for(LLMProvider.OPENAI, MockOpenAI),
        LLMProvider.ANTHROPIC: mock_for(LLMProvider.ANTHROPIC, MockAnthropic),
    }

    # Reintentos rápidos para que la demo no tarde.
    config.retry_delay = 0.1
    print(f"[MOCK activado: {mock}] No se usan API keys reales.")

    return AsyncLLMManager(config, clients=clients)


async def main():
    """Prueba el modo normal y el modo streaming del administrador."""
    try:
        config = build_config()
        manager = build_manager(config)
    except (ValueError, ValidationError) as error:
        # Configuración inválida o falta una API key: mensaje claro, sin traceback.
        print(f"Error de configuración: {error}")
        return

    print(f"Proveedor: {config.provider.value} | Modelo: {config.model}")

    try:
        print("\n--- Modo normal ---\n")
        print(await manager.generate_text(PROMPT))

        print("\n--- Modo streaming ---\n")
        async for fragment in manager.generate(PROMPT):
            print(fragment, end="", flush=True)
        print()

    except LLMError as error:
        # Error controlado (red, rate limit, API): no hay crash.
        print(f"\nError controlado: {error}")

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
