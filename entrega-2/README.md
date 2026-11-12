# Entrega 2 — Pipeline de extracción técnica (LCEL)

Pipeline asíncrono con LangChain que recibe un texto técnico (un log de error, una descripción de arquitectura) y devuelve un objeto **validado con Pydantic**: tecnologías mencionadas, nivel de criticidad y resumen técnico. Si el modelo devuelve un JSON mal formado o incompleto, la cadena **reintenta** automáticamente.

## Estructura

| Archivo | Qué hace |
|---|---|
| `schemas.py` | Esquema Pydantic `ExtraccionTecnica` (`tecnologias`, `nivel_de_criticidad`, `resumen_tecnico`). |
| `chain.py` | Modelo (`ChatOpenAI` o `ChatAnthropic`), prompt, cadena LCEL y `process_text()` async con logs. |
| `circuit_breaker.py` | Circuit breaker (CLOSED / OPEN / HALF_OPEN), el mismo de la entrega 1. |
| `mock_model.py` | Modelo falso para probar sin API keys. |
| `main.py` | Mini-script de prueba: ejecuta el pipeline sobre dos textos de ejemplo. |
| `test_chain.py` | Tests automáticos (no requieren API keys). |

## Cómo funciona

```python
prompt | model.with_structured_output(ExtraccionTecnica)   # + with_retry
```

1. `ChatPromptTemplate` le pide al modelo extraer información técnica del texto.
2. `with_structured_output(ExtraccionTecnica)` obliga al modelo a responder con el esquema y devuelve un objeto Pydantic.
3. `with_retry` reintenta (hasta 3 intentos, con espera exponencial) ante:
   - **JSON mal formado o incompleto** (`OutputParserException`, `ValidationError`),
   - **errores de red** (conexión, timeout, rate limit, errores 5xx).
4. **Circuit breaker + failover:** cada modelo (el de `LLM_PROVIDER` y el otro proveedor, si tiene API key) tiene su propio circuit breaker. Si el principal se rompe (se queda sin reintentos o falla de otra forma, por ejemplo con una API key inválida) o su circuito está abierto, la cadena sigue con el otro modelo. Cada solicitud que un modelo no logra resolver cuenta como un fallo; con 3 fallos seguidos su circuito se abre y se lo omite. Pasado el tiempo de recuperación, se vuelve a probar al principal.
5. `process_text(text)` ejecuta la cadena con `.ainvoke()` y registra en el log cada intento, cada fallo y el resultado de la validación.

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
| `LLM_PROVIDER` | `openai` o `anthropic` (por defecto `openai`). |
| `OPENAI_API_KEY` | API key de OpenAI (si el proveedor es `openai`). |
| `ANTHROPIC_API_KEY` | API key de Anthropic (si el proveedor es `anthropic`). |

La key del proveedor principal es obligatoria. La del otro proveedor es opcional: si está, se usa como modelo de respaldo.

## Ejecutar el script de prueba

```bash
python main.py
```

Ejemplo de salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Probar sin API keys (mock)

Con `LLM_MOCK` el script usa un modelo falso (`mock_model.py`). Las respuestas pasan por el parser y la validación reales, así que se ven los logs de validación y reintentos:

```bash
LLM_MOCK=ok python main.py      # responde bien a la primera
LLM_MOCK=retry python main.py   # 1º JSON cortado, 2º incompleto, 3º correcto
LLM_MOCK=failover python main.py # el principal se rompe: sigue el modelo de respaldo
LLM_MOCK=fail python main.py    # se rompen los dos modelos: falla tras 3 intentos cada uno
```

## Detalles a tener en cuenta

- La cadena es `prompt | (failover entre modelos)`. Dentro de cada modelo está `with_structured_output(ExtraccionTecnica)` envuelto con `with_retry`, para poder registrar los intentos en el log.
- Si falta la API key del proveedor principal o `LLM_PROVIDER` es inválido, `main.py` muestra un error de configuración y termina.
- Cada llamada a la API tiene un timeout de 60 segundos.

## Tests

```bash
pytest
```
