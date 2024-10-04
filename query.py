import os, shutil

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma  import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#Path a la base de datos
db_path = "db"

# Plantilla para el prompt final
PROMPT_TEMPLATE = """
You are a helpful assistant that answers the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
Answer the question in SPANISH!!
"""


# Carga del modelo para los embeddings
embedding_function = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)

# Carga de la base de datos
db = Chroma(embedding_function=embedding_function ,persist_directory= db_path, collection_name = "rag-test")

retriever = db.as_retriever(kwargs={"k": 5})

## Prompt (del usuario)
# query = "El tono que emite la unidad está muy alto. Como puedo bajar el volumen? Dame una lista paso a paso MUY DETALLADA de como bajar el volumen de la unidad"
# query = "I need to lower the volume of beeping sound of the unit, How can I lower it? Make a detailed list step by step telling how to lower the volume"

llm = ChatOllama(model="llama3.2:3b")
# llm = ChatOllama(model="llama3.1")



prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


response = retrieval_chain.invoke("Necesito bajar el volumen del tono al usar el aparato. Como puedo bajar el volumen? Dame una lista paso a paso MUY DETALLADA de como bajar el volumen de la unidad")
print(response)