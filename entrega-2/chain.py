import logging
import os
from contextvars import ContextVar
from functools import lru_cache

from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import InternalServerError as AnthropicServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import InternalServerError as OpenAIServerError
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import ValidationError

from circuit_breaker import CircuitBreaker
from schemas import ExtraccionTecnica

load_dotenv()

logger = logging.getLogger("pipeline")

DEFAULT_MODELS = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-6",
}

# Errores de red / disponibilidad: conviene reintentar.
TRANSPORT_ERRORS = (
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIRateLimitError,
    OpenAIServerError,
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicRateLimitError,
    AnthropicServerError,
)

# Errores de formato: el modelo devolvió un JSON mal formado o incompleto
# (no cumple el esquema). También se reintentan, porque en la próxima
# llamada el modelo puede responder bien.
FORMAT_ERRORS = (ValidationError, OutputParserException)

RETRYABLE_ERRORS = TRANSPORT_ERRORS + FORMAT_ERRORS

# Segundos máximos de espera por llamada a la API.
REQUEST_TIMEOUT = 60

API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class CircuitOpenError(Exception):
    """El circuito del modelo está abierto: no se le envían solicitudes."""

    def __init__(self, provider: str):
        super().__init__(f"Circuito abierto para {provider}.")
        self.provider = provider


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software senior. Extraé información "
            "técnica del texto del usuario: las tecnologías mencionadas, "
            "el nivel de criticidad (baja, media, alta o critica) y un "
            "resumen técnico breve. Usá solo lo que aparece en el texto.",
        ),
        ("human", "{text}"),
    ]
)


def _short(error: Exception) -> str:
    """Primera línea del mensaje de error, acotada, para los logs."""
    lines = str(error).splitlines() or [""]
    line = lines[0]
    return line if len(line) <= 120 else line[:117] + "..."


def _provider_name(provider: str | None) -> str:
    """Normaliza el nombre del proveedor (vacío = openai)."""
    return (provider or "").strip().lower() or "openai"


def build_model(provider: str | None = None) -> BaseChatModel:
    """
    Crea el modelo de chat según el proveedor (OpenAI o Anthropic).

    El proveedor sale del argumento o de la variable LLM_PROVIDER.
    max_retries=0 desactiva los reintentos internos del SDK para que
    el único que reintente sea with_retry (y no se multipliquen).
    El timeout evita que una llamada colgada bloquee el circuit breaker.
    """
    provider = _provider_name(provider or os.getenv("LLM_PROVIDER"))

    if provider not in API_KEY_VARS:
        raise ValueError(f"Proveedor no soportado: {provider}")

    if not os.getenv(API_KEY_VARS[provider]):
        raise ValueError(f"No se encontró {API_KEY_VARS[provider]}.")

    if provider == "openai":
        return ChatOpenAI(
            model=DEFAULT_MODELS["openai"],
            temperature=0,
            max_retries=0,
            timeout=REQUEST_TIMEOUT,
        )

    return ChatAnthropic(
        model=DEFAULT_MODELS["anthropic"],
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )


def build_models() -> dict[str, BaseChatModel]:
    """
    Crea los modelos disponibles, el principal primero.

    El principal sale de LLM_PROVIDER y es obligatorio. El otro proveedor
    se agrega como respaldo solo si tiene su API key configurada.

    Raises:
        ValueError: Si el proveedor es inválido o falta la API key principal.
    """
    primary = _provider_name(os.getenv("LLM_PROVIDER"))
    backup = "anthropic" if primary == "openai" else "openai"

    models = {primary: build_model(primary)}

    if os.getenv(API_KEY_VARS[backup]):
        models[backup] = build_model(backup)
    else:
        logger.warning(
            "Sin %s: no hay modelo de respaldo.", API_KEY_VARS[backup]
        )

    return models


# Contador de intentos de la solicitud en curso. Es una ContextVar para que
# cada solicitud (tarea) tenga el suyo, sin estado compartido ni fugas.
_attempt_counter: ContextVar[list[int]] = ContextVar("attempt_counter")


def _build_provider_chain(
    name: str,
    model: BaseChatModel,
    max_attempts: int,
    initial_wait: float,
) -> Runnable:
    """
    Arma la cadena de un proveedor: salida estructurada + reintentos con
    backoff exponencial ante errores de red y de formato.
    """
    structured_model = model.with_structured_output(ExtraccionTecnica)

    async def logged_call(prompt_value, config: RunnableConfig):
        """Llama al modelo y registra cada intento y su resultado."""
        counter = _attempt_counter.get()
        counter[0] += 1
        attempt = counter[0]
        logger.info(
            "[%s] Llamando al modelo (intento %d/%d)", name, attempt, max_attempts
        )

        try:
            result = await structured_model.ainvoke(prompt_value, config)

            # Algunos modelos devuelven None cuando no generan la estructura.
            if not isinstance(result, ExtraccionTecnica):
                raise OutputParserException(
                    "El modelo no devolvió la estructura esperada."
                )

            return result

        except RETRYABLE_ERRORS as error:
            kind = "formato" if isinstance(error, FORMAT_ERRORS) else "red"
            logger.warning(
                "[%s] Intento %d falló (%s) - %s: %s",
                name,
                attempt,
                kind,
                type(error).__name__,
                _short(error),
            )
            raise

    return RunnableLambda(logged_call).with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        stop_after_attempt=max_attempts,
        exponential_jitter_params={
            "initial": initial_wait,
            "jitter": initial_wait,
        },
    )


def build_chain(
    models: dict[str, BaseChatModel],
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    breakers: dict[str, CircuitBreaker] | None = None,
) -> Runnable:
    """
    Arma la cadena: prompt | modelo con salida estructurada, con reintentos,
    circuit breaker por modelo y failover entre modelos.

    Se prueba el primer modelo (con sus reintentos). Si se queda sin intentos,
    falla de cualquier otra forma (API key inválida, rechazo, etc.) o su
    circuito está abierto, se sigue con el siguiente modelo. Cada solicitud
    que un modelo no logra resolver cuenta como un fallo para su circuito.

    Args:
        models: Modelos por nombre, en orden de prioridad (principal primero).
        max_attempts: Intentos máximos por modelo ante un error recuperable.
        initial_wait: Espera inicial (segundos) del backoff exponencial.
        failure_threshold: Solicitudes fallidas consecutivas para abrir el circuito.
        recovery_timeout: Segundos hasta volver a probar un modelo caído.
        breakers: Circuit breakers ya creados por nombre (opcional).

    Raises:
        ValueError: Si no se indica ningún modelo.
    """
    if not models:
        raise ValueError("Se necesita al menos un modelo.")

    breakers = breakers if breakers is not None else {}

    provider_chains = []
    for name, model in models.items():
        breaker = breakers.setdefault(
            name,
            CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            ),
        )
        provider_chains.append(
            (
                name,
                breaker,
                _build_provider_chain(name, model, max_attempts, initial_wait),
            )
        )

    async def with_failover(prompt_value, config: RunnableConfig):
        """Prueba cada modelo en orden hasta que uno responda."""
        first_error = None
        circuit_error = None

        for index, (name, breaker, provider_chain) in enumerate(provider_chains):

            if not await breaker.allow_request():
                logger.warning("[%s] Circuito abierto: se omite.", name)
                circuit_error = circuit_error or CircuitOpenError(name)

            else:
                # Cada modelo arranca con su propio conteo de intentos.
                _attempt_counter.set([0])

                try:
                    result = await provider_chain.ainvoke(prompt_value, config)
                    await breaker.record_success()
                    return result

                except Exception as error:
                    await breaker.record_failure()
                    logger.warning(
                        "[%s] No pudo resolver la solicitud (%s).",
                        name,
                        type(error).__name__,
                    )
                    first_error = first_error or error

                except BaseException:
                    # Cancelación: no dejamos trabada la prueba HALF_OPEN.
                    await breaker.release_probe()
                    raise

            if index + 1 < len(provider_chains):
                logger.warning(
                    "Pasando al modelo de respaldo: %s",
                    provider_chains[index + 1][0],
                )

        # Se prefiere informar el error real antes que el del circuito.
        raise first_error or circuit_error

    return PROMPT | RunnableLambda(with_failover)


@lru_cache(maxsize=1)
def get_default_chain() -> Runnable:
    """Cadena por defecto (se crea una sola vez, la primera vez que se usa)."""
    return build_chain(build_models())


async def process_text(
    text: str,
    chain: Runnable | None = None,
) -> ExtraccionTecnica:
    """
    Extrae información técnica de un texto.

    Args:
        text: Texto a analizar (log de error, descripción de arquitectura...).
        chain: Cadena a usar. Si se omite, se usa la cadena por defecto.

    Returns:
        Objeto ExtraccionTecnica ya validado.

    Raises:
        ValueError: Si el texto está vacío.
        ValidationError / OutputParserException: Si tras todos los
            reintentos el modelo no respetó el esquema.
        Exception de red/API: Si fallaron todos los modelos disponibles
            (se informa el primer error real).
        CircuitOpenError: Si los circuitos de todos los modelos están abiertos.
    """
    if not text or not text.strip():
        raise ValueError("El texto no puede estar vacío.")

    chain = chain or get_default_chain()

    logger.info("Procesando texto (%d caracteres)...", len(text))

    try:
        result = await chain.ainvoke({"text": text})
    except FORMAT_ERRORS as error:
        logger.error(
            "Validación fallida tras los reintentos: %s: %s",
            type(error).__name__,
            _short(error),
        )
        raise
    except Exception as error:
        logger.error(
            "Falló la llamada tras los reintentos: %s: %s",
            type(error).__name__,
            _short(error),
        )
        raise

    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(result.tecnologias),
        result.nivel_de_criticidad,
    )
    return result
