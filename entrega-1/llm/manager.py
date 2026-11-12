
import asyncio
import os

from typing import AsyncIterator

import openai
import anthropic

from dotenv import load_dotenv

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from schemas import LLMConfig, LLMProvider

from llm.circuit_breaker import CircuitBreaker
from llm.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMProviderError,
    LLMStreamInterruptedError,
)


load_dotenv()


class AsyncLLMManager:
    """
    Administra múltiples proveedores LLM con streaming, reintentos,
    circuit breaker y failover automático.
    """

    def __init__(self, config: LLMConfig, clients=None):
        """
        Inicializa los clientes y los circuit breakers de los proveedores.

        Args:
            config: Configuración validada del administrador.
            clients: Clientes ya creados por proveedor (por ejemplo, mocks
                para probar sin API keys). Si se omite, se crean los
                clientes reales leyendo las API keys del entorno.
        """
        self.config = config

        # El proveedor configurado es siempre el proveedor principal.
        self.primary_provider = config.provider

        # Si no se especifica un proveedor alternativo, utilizamos el otro.
        self.fallback_provider = (
            config.fallback_provider
            or (
                LLMProvider.ANTHROPIC
                if config.provider == LLMProvider.OPENAI
                else LLMProvider.OPENAI
            )
        )

        if self.fallback_provider == self.primary_provider:
            raise ValueError(
                "El proveedor alternativo debe ser diferente del principal."
            )

        if clients is not None:
            # Clientes inyectados: no se necesitan API keys.
            self.clients = dict(clients)
        else:
            self.clients = self._create_clients()

        # Cada proveedor mantiene su propio circuit breaker.
        self.circuit_breakers = {
            provider: CircuitBreaker(
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout,
            )
            for provider in self.clients
        }

    def _create_clients(self):
        """Crea los clientes reales a partir de las API keys del entorno."""
        # El proveedor principal es obligatorio.
        clients = {
            self.primary_provider: self._initialize_client(
                self.primary_provider
            )
        }

        # El proveedor alternativo es opcional: sin API key no estará disponible.
        fallback_client = self._initialize_client(
            self.fallback_provider,
            required=False,
        )

        if fallback_client is not None:
            clients[self.fallback_provider] = fallback_client

        return clients

    def _initialize_client(self, provider, required=True):
        """
        Inicializa el cliente correspondiente a un proveedor.

        Args:
            provider: Proveedor que se desea inicializar.
            required: Indica si la API key es obligatoria.

        Returns:
            Cliente asíncrono o None si el proveedor opcional no está configurado.

        Raises:
            ValueError: Si falta una API key obligatoria.
        """
        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                if required:
                    raise ValueError(
                        "No se encontró OPENAI_API_KEY."
                    )
                return None

            return AsyncOpenAI(
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=0,
            )

        if provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                if required:
                    raise ValueError(
                        "No se encontró ANTHROPIC_API_KEY."
                    )
                return None

            return AsyncAnthropic(
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=0,
            )

        raise ValueError(f"Proveedor no soportado: {provider}")

    def _get_model(self, provider):
        """
        Devuelve el modelo correspondiente al proveedor.

        Permite configurar un modelo alternativo independiente.
        """
        if provider == self.primary_provider:
            return self.config.model

        if self.config.fallback_model:
            return self.config.fallback_model

        # Modelos alternativos predeterminados.
        if provider == LLMProvider.OPENAI:
            return "gpt-4.1"

        return "claude-sonnet-4-6"

    def _build_provider_params(self, provider):
        """
        Construye parámetros compatibles con el proveedor seleccionado.

        Anthropic no recibe temperature ni top_p.
        """
        params = {
            "model": self._get_model(provider),
            "max_tokens": self.config.max_tokens,
        }

        if provider == LLMProvider.OPENAI:
            if self.config.temperature is not None:
                params["temperature"] = self.config.temperature

            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p

        return params

    def _is_retryable_error(self, error):
        """
        Determina si un error puede recuperarse mediante un reintento.

        Contempla errores de conexión, timeout, rate limiting y errores 5xx.
        """
        retryable_types = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )

        if isinstance(error, retryable_types):
            return True

        # Algunos errores HTTP requieren inspeccionar su código de estado.
        if isinstance(error, (openai.APIStatusError, anthropic.APIStatusError)):
            return error.status_code == 429 or error.status_code >= 500

        return False

    def _build_controlled_error(self, error, provider):
        """
        Convierte excepciones de los SDK en errores propios del administrador.
        """
        message = str(error)
        provider_name = provider.value

        if isinstance(
            error,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ),
        ):
            return LLMConnectionError(
                f"Error de conexión con {provider_name}: {message}",
                provider_name,
            )

        if isinstance(
            error,
            (openai.RateLimitError, anthropic.RateLimitError),
        ):
            return LLMRateLimitError(
                f"Rate limit de {provider_name}: {message}",
                provider_name,
            )

        return LLMProviderError(
            f"Error del proveedor {provider_name}: {message}",
            provider_name,
        )


    def _build_request(self, provider, prompt, stream):
        """
        Arma los argumentos de la solicitud según el proveedor.

        Args:
            provider: Proveedor de destino.
            prompt: Texto enviado por el usuario.
            stream: Indica si la respuesta se recibe por streaming.
        """
        params = self._build_provider_params(provider)
        request = {
            **params,
            "messages": [{"role": "user", "content": prompt}],
        }

        if stream:
            request["stream"] = True

        if provider == LLMProvider.OPENAI:
            # Los modelos nuevos de OpenAI usan max_completion_tokens.
            request["max_completion_tokens"] = request.pop("max_tokens")

        return request

    async def _stream_from_provider(
        self,
        provider,
        prompt,
    ) -> AsyncIterator[str]:
        """
        Ejecuta una solicitud de streaming contra un proveedor específico.
        """
        client = self.clients[provider]
        request = self._build_request(provider, prompt, stream=True)

        if provider == LLMProvider.OPENAI:
            stream = await client.chat.completions.create(**request)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        elif provider == LLMProvider.ANTHROPIC:
            stream = await client.messages.create(**request)

            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield event.delta.text

        else:
            raise ValueError(
                f"Proveedor no soportado: {provider}"
            )

    async def _complete_from_provider(self, provider, prompt) -> str:
        """
        Ejecuta una solicitud normal (sin streaming) contra un proveedor.

        Returns:
            Texto completo de la respuesta.
        """
        client = self.clients[provider]
        request = self._build_request(provider, prompt, stream=False)

        if provider == LLMProvider.OPENAI:
            response = await client.chat.completions.create(**request)
            return response.choices[0].message.content or ""

        if provider == LLMProvider.ANTHROPIC:
            response = await client.messages.create(**request)
            return "".join(
                block.text for block in response.content
                if block.type == "text"
            )

        raise ValueError(f"Proveedor no soportado: {provider}")

    @staticmethod
    def _validate_prompt(prompt):
        """Valida que el prompt no esté vacío."""
        if not isinstance(prompt, str):
            raise ValueError("El prompt debe ser un texto.")

        if not prompt.strip():
            raise ValueError("El prompt no puede estar vacío.")

    async def _wait_before_retry(self, breaker, attempt) -> bool:
        """
        Espera con backoff exponencial antes del próximo reintento.

        Returns:
            True si corresponde reintentar con el mismo proveedor,
            False si hay que pasar al siguiente proveedor.
        """
        if attempt >= self.config.max_retries:
            return False

        # Si el circuito ya se abrió, no tiene sentido esperar.
        if await breaker.get_state() == "OPEN":
            return False

        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        # Si se abrió mientras esperábamos, dejamos de insistir.
        return await breaker.get_state() != "OPEN"

    async def generate_text(self, prompt: str) -> str:
        """
        Genera una respuesta completa (modo normal, sin streaming).

        Reintenta los errores recuperables y, si el proveedor principal
        sigue fallando, cambia al proveedor alternativo.

        Raises:
            ValueError: Si el prompt está vacío.
            LLMError: Ante errores controlados de red, rate limit o API.
        """
        self._validate_prompt(prompt)

        last_error = None
        last_provider = None

        for provider in (self.primary_provider, self.fallback_provider):

            if provider not in self.clients:
                continue

            breaker = self.circuit_breakers[provider]

            if not await breaker.allow_request():
                continue

            for attempt in range(self.config.max_retries + 1):
                try:
                    text = await self._complete_from_provider(
                        provider,
                        prompt,
                    )
                    await breaker.record_success()
                    return text

                except Exception as error:
                    if not self._is_retryable_error(error):
                        await breaker.release_probe()
                        raise self._build_controlled_error(
                            error,
                            provider,
                        ) from error

                    last_error = error
                    last_provider = provider
                    await breaker.record_failure()

                    if not await self._wait_before_retry(breaker, attempt):
                        break

                except BaseException:
                    # Cancelación: no dejamos trabada la prueba HALF_OPEN.
                    await breaker.release_probe()
                    raise

        if last_error is not None:
            raise self._build_controlled_error(
                last_error,
                last_provider,
            ) from last_error

        raise LLMProviderError(
            "No hay proveedores disponibles para generar la respuesta."
        )

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        """
        Genera una respuesta utilizando streaming y failover automático.

        Reintenta los errores recuperables antes de cambiar de proveedor.

        Si ya se entregaron fragmentos al consumidor, no reinicia la respuesta
        con otro proveedor para evitar duplicar contenido.
        """
        self._validate_prompt(prompt)

        last_error = None
        last_provider = None

        for provider in (self.primary_provider, self.fallback_provider):

            # Si no existe cliente, pasamos al siguiente proveedor.
            if provider not in self.clients:
                continue

            breaker = self.circuit_breakers[provider]

            if not await breaker.allow_request():
                continue

            for attempt in range(self.config.max_retries + 1):

                received_content = False

                try:
                    async for fragment in self._stream_from_provider(
                        provider,
                        prompt,
                    ):
                        if fragment:
                            received_content = True
                            yield fragment

                    # Si la generación finalizó correctamente,
                    # restablecemos el circuito.
                    await breaker.record_success()
                    return

                except Exception as error:

                    if not self._is_retryable_error(error):
                        await breaker.release_probe()
                        raise self._build_controlled_error(
                            error,
                            provider,
                        ) from error

                    last_error = error
                    last_provider = provider
                    await breaker.record_failure()

                    # No podemos reiniciar una respuesta parcialmente enviada.
                    if received_content:
                        raise LLMStreamInterruptedError(
                            "El streaming se interrumpió después de "
                            "entregar contenido. No es seguro reiniciarlo "
                            "automáticamente.",
                            provider.value,
                        ) from error

                    if not await self._wait_before_retry(breaker, attempt):
                        break

                except BaseException:
                    # El consumidor cortó el stream o se canceló la tarea:
                    # no dejamos trabada la prueba HALF_OPEN.
                    await breaker.release_probe()
                    raise

        if last_error is not None:
            raise self._build_controlled_error(
                last_error,
                last_provider,
            ) from last_error

        raise LLMProviderError(
            "No hay proveedores disponibles para generar la respuesta."
        )

    async def close(self):
        """
        Cierra todos los clientes HTTP inicializados.
        """
        # Si un cliente falla al cerrarse, igual se cierran los demás.
        await asyncio.gather(
            *(client.close() for client in self.clients.values()),
            return_exceptions=True,
        )
