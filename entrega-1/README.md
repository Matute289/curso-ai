# Entrega 1 — Administrador asíncrono de LLMs

Cliente asíncrono para **OpenAI** y **Anthropic** con validación de parámetros (Pydantic), streaming, reintentos, circuit breaker y failover entre proveedores.

## Estructura

| Archivo | Qué hace |
|---|---|
| `schemas.py` | Esquema Pydantic (`LLMConfig`) que valida los parámetros (temperatura de 0 a 2, `max_tokens > 0`, etc.). |
| `llm/manager.py` | `AsyncLLMManager`: elige el proveedor según la configuración. `generate_text()` devuelve la respuesta completa; `generate()` entrega fragmentos con `yield` (streaming). |
| `llm/circuit_breaker.py` | Circuit breaker por proveedor (CLOSED / OPEN / HALF_OPEN). |
| `llm/mock_clients.py` | Clientes falsos de OpenAI y Anthropic para probar sin API keys. |
| `llm/exceptions.py` | Errores controlados: conexión, rate limit, proveedor, streaming interrumpido. |
| `main.py` | Script de prueba: ejecuta el modo normal y el modo streaming. |
| `test_*.py` | Tests automáticos (no requieren API keys). |

## Requisitos

- Python 3.14

## Instalación

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completar las claves
```

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | Proveedor principal: `openai` o `anthropic` (por defecto `openai`). |
| `OPENAI_API_KEY` | API key de OpenAI. |
| `ANTHROPIC_API_KEY` | API key de Anthropic. |

La key del proveedor principal es obligatoria. La del otro proveedor es opcional: si está, se usa como respaldo (failover) cuando el principal falla.

## Ejecutar el script de prueba

```bash
python main.py
```

Imprime la respuesta en modo normal y luego la misma consulta en modo streaming (los fragmentos aparecen a medida que llegan). Para probar con el otro proveedor:

```bash
LLM_PROVIDER=anthropic python main.py
```

## Probar sin API keys (mock)

Con `LLM_MOCK` el script usa clientes falsos (`llm/mock_clients.py`) y no necesita claves ni internet:

```bash
LLM_MOCK=1 python main.py      # respuestas normales simuladas
LLM_MOCK=fail python main.py   # el proveedor principal falla por rate limit y se activa el failover
```

## Manejo de errores

Los errores de red, timeout, rate limit y de la API se convierten en excepciones propias (`LLMConnectionError`, `LLMRateLimitError`, `LLMProviderError`, todas subclases de `LLMError`) en lugar de romper el programa. Los errores recuperables se reintentan con espera exponencial y luego se pasa al proveedor alternativo. Si el streaming ya entregó texto y se corta, se lanza `LLMStreamInterruptedError` para no duplicar contenido.

## Detalles a tener en cuenta

- **Qué dispara el failover:** errores de conexión, timeout, rate limit (429) y errores 5xx. Un error como API key inválida (401) o modelo inexistente (404) se informa directamente como error controlado, sin pasar al otro proveedor.
- **Temperatura y top_p:** se envían solo a OpenAI. Con Anthropic se validan pero se ignoran.
- **Errores de configuración:** si falta la API key del proveedor principal o `LLM_PROVIDER` es inválido, `main.py` muestra un mensaje de error y termina.

## Tests

```bash
pytest
```
