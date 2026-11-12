import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lcel_async import build_chain


def test_la_cadena_devuelve_texto_plano():
    chain = build_chain(FakeListChatModel(responses=["Hola, soy un mock."]))

    result = asyncio.run(chain.ainvoke({"pregunta": "¿Qué es LCEL?"}))

    assert isinstance(result, str)
    assert result == "Hola, soy un mock."
