from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMProvider(str, Enum):
    """Define los proveedores de modelos soportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    """
    Define y valida los parámetros de configuración de un modelo LLM.

    Permite utilizar una configuración común para OpenAI y Anthropic,
    contemplando las diferencias entre los parámetros de cada proveedor.
    """

    fallback_provider: Optional[LLMProvider] = None

    # Modelo que se utilizará cuando se active el failover.
    fallback_model: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Proveedor que se utilizará para generar las respuestas.
    provider: LLMProvider

    # Identificador del modelo correspondiente al proveedor seleccionado.
    model: str = Field(
        min_length=1,
        description="Nombre o identificador del modelo."
    )

    # Cantidad máxima de tokens que puede generar el modelo.
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Cantidad máxima de tokens de salida."
    )

    # Temperatura: parámetro soportado por OpenAI.
    # Anthropic no utiliza este parámetro, por lo que se deja opcional.
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperatura de generación. Solo se utiliza con OpenAI."
    )

    # Controla la diversidad de selección de tokens.
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Parámetro de muestreo opcional."
    )

    # Tiempo máximo de espera de una solicitud, expresado en segundos.
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Tiempo máximo de espera de la API."
    )

    # Cantidad máxima de reintentos ante errores recuperables.
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Cantidad máxima de reintentos."
    )

    # Cantidad de errores consecutivos antes de abrir el circuito.
    failure_threshold: int = Field(
        default=3,
        ge=1,
    )

    # Tiempo en segundos antes de volver a probar un proveedor.
    recovery_timeout: float = Field(
        default=30.0,
        gt=0,
    )

    # Tiempo inicial de espera entre reintentos.
    retry_delay: float = Field(
        default=1.0,
        gt=0,
    )
