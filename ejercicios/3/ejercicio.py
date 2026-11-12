from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model  = ChatOpenAI(model="gpt-4o")  # el LLM
parser = StrOutputParser()  # convierte la respuesta en str

# Datos de entrada: una lista de reseñas (str)
reviews = ["Llegó rápido y funciona perfecto",
           "Se rompió a la semana, pésima calidad",
           "El envío tardó, pero el producto está bien"]

# Cadena 1: clasifica el sentimiento de una reseña
sentiment_chain = (
    ChatPromptTemplate.from_template("...sentimiento...: {review}") | model | parser)
# Cadena 2: extrae el tema principal de la misma reseña
topic_chain = (
    ChatPromptTemplate.from_template("...tema principal...: {review}") | model | parser)

# Paso paralelo: ambas cadenas reciben la MISMA
# reseña y corren a la vez.
# Devuelve un dict: {"sentiment": ..., "topic": ...}
analysis = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

# Prompt final: {sentiment} y {topic} coinciden
# con las claves del dict anterior
ticket_prompt = ChatPromptTemplate.from_template(
    "Sentimiento: {sentiment}\nTema: {topic}\nEscribí una línea de ticket...")

# Cadena completa: reseña → (sentimiento ∥ tema)
# → prompt → modelo → texto
full_chain = analysis | ticket_prompt | model | parser

# batch recibe una LISTA de entradas y ejecuta
# full_chain una vez por cada una, en paralelo.
# Cada entrada es un dict con la clave "review"
# (la variable que esperan las cadenas 1 y 2).
inputs = [{"review": r} for r in reviews]

# max_concurrency=5: máximo 5 cadenas a la vez
# (evita rate limits de la API)
tickets = full_chain.batch(inputs, config={"max_concurrency": 5})

# El resultado conserva el orden: tickets[i] corresponde a reviews[i]
for r, t in zip(reviews, tickets):
    print(f"{r}  →  {t}")