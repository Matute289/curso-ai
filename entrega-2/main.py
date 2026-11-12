import asyncio
import logging
import os

from chain import build_chain, build_models, process_text
from mock_model import build_mock_model

SAMPLES = [
    (
        "Nuestra API en FastAPI usa Redis como caché y PostgreSQL para "
        "persistir los datos. En horas pico, el pool de conexiones a "
        "PostgreSQL se agota y los usuarios reciben errores 500."
    ),
    (
        "[2026-11-10 14:03:12] ERROR kafka.consumer: Offset commit failed, "
        "rebalance in progress. El servicio de pagos en Java (Spring Boot) "
        "dejó de procesar mensajes hace 20 minutos."
    ),
]


# Escenarios del mock: (modelo principal, modelo de respaldo).
MOCK_SCENARIOS = {
    "ok": ("ok", "backup"),            # el principal responde bien
    "retry": ("retry", "backup"),      # el principal se recupera reintentando
    "failover": ("fail", "backup"),    # el principal se rompe: sigue el respaldo
    "fail": ("fail", "fail"),          # se rompen los dos modelos
}


def build_demo_chain():
    """
    Crea la cadena a usar. Con LLM_MOCK activo usa modelos falsos
    (sin API keys): ok | retry | failover | fail.
    """
    scenario = os.getenv("LLM_MOCK", "").lower()

    if not scenario:
        return build_chain(build_models())

    scenario = "ok" if scenario == "1" else scenario
    primary, backup = MOCK_SCENARIOS[scenario]
    print(f"[MOCK activado: {scenario}] No se usan API keys reales.\n")

    models = {
        "mock-principal": build_mock_model(primary),
        "mock-respaldo": build_mock_model(backup),
    }

    # Espera mínima entre reintentos para que la demo no tarde.
    # failure_threshold=1: el circuito se abre con la primera solicitud
    # fallida, para ver cómo se omite el modelo roto en la siguiente.
    return build_chain(models, initial_wait=0.1, failure_threshold=1)


async def main():
    """Ejecuta el pipeline de extracción sobre textos de ejemplo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        chain = build_demo_chain()
    except ValueError as error:
        # Proveedor inválido o falta la API key: mensaje claro, sin traceback.
        print(f"Error de configuración: {error}")
        return

    for text in SAMPLES:
        print(f"Texto: {text}\n")

        try:
            result = await process_text(text, chain=chain)
            print(result.model_dump_json(indent=2))
        except Exception as error:
            print(f"Error: {type(error).__name__}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
