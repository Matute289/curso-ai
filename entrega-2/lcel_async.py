"""
Ejercicio opcional: refactorización a LCEL asíncrono.

Cadena: ChatPromptTemplate | modelo | StrOutputParser, ejecutada con
`await chain.ainvoke({...})`. Devuelve la respuesta como texto plano.

Uso:
    python lcel_async.py                # modelo real (según LLM_PROVIDER)
    LLM_MOCK=1 python lcel_async.py     # modelo falso, sin API keys
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

load_dotenv()

# Prompt con roles definidos (System / Human) y una variable: {pregunta}.
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Sos un asistente técnico. Respondé en castellano y de forma breve."),
        ("human", "{pregunta}"),
    ]
)


def build_model() -> BaseChatModel:
    """Crea el modelo según LLM_PROVIDER (openai por defecto) o un mock."""
    if os.getenv("LLM_MOCK", "").lower() not in ("", "0", "false", "no"):
        return FakeListChatModel(
            responses=["LCEL es una forma declarativa de armar cadenas con el operador |."]
        )

    provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

    if provider == "anthropic":
        return ChatAnthropic(model="claude-sonnet-4-6")

    # Parámetros base: modelo y temperatura.
    return ChatOpenAI(model="gpt-4.1", temperature=0.7)


def build_chain(model: BaseChatModel | None = None) -> Runnable:
    """Arma la cadena LCEL: prompt | modelo | parser."""
    return PROMPT | (model or build_model()) | StrOutputParser()


async def main():
    """Ejecuta la cadena de forma asíncrona e imprime la respuesta."""
    try:
        chain = build_chain()
    except Exception as error:
        # Por ejemplo, falta la API key del proveedor.
        print(f"Error de configuración: {error}")
        return

    # Las claves del dict deben coincidir con las variables del prompt.
    respuesta = await chain.ainvoke({"pregunta": "¿Qué es LCEL en LangChain?"})

    # Texto plano (str), no un AIMessage.
    print(f"¿Es texto plano? {isinstance(respuesta, str)}")
    print(respuesta)


if __name__ == "__main__":
    asyncio.run(main())
