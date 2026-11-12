import asyncio
from types import SimpleNamespace

import openai
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from circuit_breaker import CircuitBreaker
from chain import CircuitOpenError, build_chain, process_text
from mock_model import BAD_JSON, GOOD_JSON, MockChatModel, build_mock_model
from schemas import ExtraccionTecnica


def run(coro):
    return asyncio.run(coro)


def make_chain(scenario, backup="backup", max_attempts=3, **kwargs):
    models = {"principal": build_mock_model(scenario)}
    if backup:
        models["respaldo"] = build_mock_model(backup)
    return build_chain(
        models, max_attempts=max_attempts, initial_wait=0.01, **kwargs
    )


# --- Esquema ---

def test_esquema_valido():
    data = ExtraccionTecnica(
        tecnologias=["FastAPI"],
        nivel_de_criticidad="alta",
        resumen_tecnico="API con cuello de botella.",
    )
    assert data.nivel_de_criticidad == "alta"


def test_esquema_rechaza_criticidad_invalida():
    with pytest.raises(ValidationError):
        ExtraccionTecnica(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="altisima",
            resumen_tecnico="API con cuello de botella.",
        )


def test_esquema_rechaza_tecnologias_vacias():
    with pytest.raises(ValidationError):
        ExtraccionTecnica(
            tecnologias=[],
            nivel_de_criticidad="alta",
            resumen_tecnico="API con cuello de botella.",
        )


# --- Cadena ---

def test_respuesta_correcta_a_la_primera():
    result = run(process_text("API en FastAPI con Redis", chain=make_chain("ok")))

    assert isinstance(result, ExtraccionTecnica)
    assert result.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert result.nivel_de_criticidad == "alta"


def test_reintenta_ante_json_mal_formado_o_incompleto():
    # Respuestas: JSON cortado -> JSON incompleto -> JSON correcto.
    result = run(process_text("API en FastAPI", chain=make_chain("retry")))

    assert isinstance(result, ExtraccionTecnica)
    assert result.nivel_de_criticidad == "alta"


def test_falla_tras_agotar_los_reintentos():
    with pytest.raises((ValidationError, OutputParserException)):
        run(process_text("API en FastAPI", chain=make_chain("fail", backup="fail")))


def test_texto_vacio():
    with pytest.raises(ValueError):
        run(process_text("   ", chain=make_chain("ok")))


def test_reintenta_ante_error_de_red():
    """Un rate limit transitorio se reintenta y luego responde bien."""
    calls = {"n": 0}
    expected = ExtraccionTecnica(
        tecnologias=["Redis"],
        nivel_de_criticidad="media",
        resumen_tecnico="Caché en Redis con latencia alta.",
    )

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            response = SimpleNamespace(request=None, status_code=429, headers={})
            raise openai.RateLimitError("rate limit", response=response, body=None)
        return expected

    class FlakyModel(MockChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(flaky)

    chain = build_chain(
        {"principal": FlakyModel(responses=["x"])},
        max_attempts=3,
        initial_wait=0.01,
    )

    assert run(process_text("Redis lento", chain=chain)) == expected
    assert calls["n"] == 3


# --- Circuit breaker y failover ---

def test_failover_si_el_modelo_principal_se_rompe():
    """Si el principal falla siempre, responde el modelo de respaldo."""
    result = run(process_text("Servicio caído", chain=make_chain("fail")))

    assert result.tecnologias == ["Kubernetes"]
    assert "respaldo" in result.resumen_tecnico


def test_el_circuito_se_abre_y_se_omite_el_modelo_roto():
    breakers = {}
    chain = make_chain("fail", breakers=breakers, failure_threshold=1)

    run(process_text("Servicio caído", chain=chain))
    assert run(breakers["principal"].get_state()) == "OPEN"
    assert run(breakers["respaldo"].get_state()) == "CLOSED"

    # Segunda solicitud: el principal está abierto, responde el respaldo.
    result = run(process_text("Otro texto", chain=chain))
    assert result.tecnologias == ["Kubernetes"]


def test_falla_si_se_rompen_los_dos_modelos():
    with pytest.raises((ValidationError, OutputParserException)):
        run(process_text("Servicio caído", chain=make_chain("fail", backup="fail")))


def test_circuito_abierto_en_todos_los_modelos():
    breakers = {}
    chain = make_chain(
        "fail",
        backup="fail",
        breakers=breakers,
        failure_threshold=1,
        recovery_timeout=60,
    )

    with pytest.raises((ValidationError, OutputParserException)):
        run(process_text("Servicio caído", chain=chain))

    # Ahora ambos circuitos están abiertos: no se llama a ningún modelo.
    with pytest.raises(CircuitOpenError):
        run(process_text("Otro texto", chain=chain))


def test_el_modelo_principal_se_recupera_tras_el_timeout():
    """Pasado recovery_timeout, se vuelve a probar al principal."""
    breakers = {}
    primary = MockChatModel(responses=[BAD_JSON, BAD_JSON, BAD_JSON, GOOD_JSON])
    chain = build_chain(
        {"principal": primary, "respaldo": build_mock_model("backup")},
        initial_wait=0.01,
        failure_threshold=1,
        recovery_timeout=0.2,
        breakers=breakers,
    )

    run(process_text("texto", chain=chain))           # el principal se rompe
    assert run(breakers["principal"].get_state()) == "OPEN"

    async def later():
        await asyncio.sleep(0.3)
        return await process_text("texto", chain=chain)

    result = run(later())                             # vuelve a probarse
    assert result.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert run(breakers["principal"].get_state()) == "CLOSED"


def test_failover_ante_error_no_recuperable():
    """Una API key inválida (401) no se reintenta, pero sigue el respaldo."""

    def auth_error(_):
        response = SimpleNamespace(request=None, status_code=401, headers={})
        raise openai.AuthenticationError("key inválida", response=response, body=None)

    class BrokenModel(MockChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(auth_error)

    chain = build_chain(
        {
            "principal": BrokenModel(responses=["x"]),
            "respaldo": build_mock_model("backup"),
        },
        initial_wait=0.01,
    )

    result = run(process_text("texto", chain=chain))
    assert result.tecnologias == ["Kubernetes"]


def test_resultado_none_se_trata_como_fallo():
    """Si el modelo devuelve None (sin estructura), se reintenta/failover."""

    class NoneModel(MockChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: None)

    chain = build_chain(
        {
            "principal": NoneModel(responses=["x"]),
            "respaldo": build_mock_model("backup"),
        },
        initial_wait=0.01,
    )

    result = run(process_text("texto", chain=chain))
    assert result.tecnologias == ["Kubernetes"]


def test_sin_modelos():
    with pytest.raises(ValueError):
        build_chain({})


def test_se_informa_el_error_real_y_no_el_del_circuito():
    """Si el respaldo tiene el circuito abierto, se informa el error real."""
    breakers = {"respaldo": CircuitBreaker(failure_threshold=1, recovery_timeout=60)}
    run(breakers["respaldo"].record_failure())  # abre el circuito del respaldo

    chain = build_chain(
        {
            "principal": build_mock_model("fail"),
            "respaldo": build_mock_model("backup"),
        },
        initial_wait=0.01,
        breakers=breakers,
    )

    with pytest.raises((ValidationError, OutputParserException)):
        run(process_text("texto", chain=chain))
