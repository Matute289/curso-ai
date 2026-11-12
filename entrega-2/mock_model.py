"""
Modelo de chat falso para probar la cadena sin API keys.

Devuelve respuestas JSON predefinidas, una por llamada. La salida pasa por
el parser y la validación Pydantic reales, así que un JSON mal formado o
incompleto dispara los mismos errores (y reintentos) que con un modelo real.
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import PydanticOutputParser

BAD_JSON = '{"tecnologias": ["FastAPI", "Redis"], "nivel_de_criticidad": "alta"'
INCOMPLETE_JSON = '{"tecnologias": [], "nivel_de_criticidad": "alta"}'
GOOD_JSON = (
    '{"tecnologias": ["FastAPI", "Redis", "PostgreSQL"], '
    '"nivel_de_criticidad": "alta", '
    '"resumen_tecnico": "API con caché en Redis y persistencia en '
    'PostgreSQL; cuello de botella en conexiones concurrentes."}'
)
BACKUP_JSON = (
    '{"tecnologias": ["Kubernetes"], "nivel_de_criticidad": "media", '
    '"resumen_tecnico": "(Respuesta del modelo de respaldo) Servicio '
    'desplegado en Kubernetes con degradación parcial."}'
)


class MockChatModel(FakeListChatModel):
    """Modelo falso con soporte para with_structured_output."""

    def with_structured_output(self, schema, **kwargs):
        """Parsea y valida la respuesta contra el esquema, como el real."""
        return self | PydanticOutputParser(pydantic_object=schema)


def build_mock_model(scenario: str = "ok") -> MockChatModel:
    """
    Crea un modelo falso según el escenario.

    Escenarios:
        ok:     responde bien a la primera.
        retry:  1º JSON mal formado, 2º JSON incompleto, 3º correcto.
        fail:   siempre responde JSON mal formado (modelo "roto").
        backup: responde bien, con un texto que identifica al respaldo.
    """
    scenarios = {
        "ok": [GOOD_JSON],
        "retry": [BAD_JSON, INCOMPLETE_JSON, GOOD_JSON],
        "fail": [BAD_JSON],
        "backup": [BACKUP_JSON],
    }

    if scenario not in scenarios:
        raise ValueError(f"Escenario de mock desconocido: {scenario}")

    return MockChatModel(responses=scenarios[scenario])
