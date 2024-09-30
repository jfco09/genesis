import os, shutil

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma  import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage

#Path a la base de datos
db_path = "db"

# Plantilla para el prompt
PROMPT_TEMPLATE = """
You are a helpful assistant that answers the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Carga del modelo para los embeddings
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Carga de la base de datos
db = Chroma(embedding_function=embedding_function ,persist_directory= db_path, collection_name = "rag-test")

## Prompt (del usuario)
# query = "El tono que emite la unidad está muy alto. Como puedo bajar el volumen? Dame una lista paso a paso MUY DETALLADA de como bajar el volumen de la unidad"
query = "I need to lower the volume of beeping sound of the unit, How can I lower it? Make a detailed list step by step telling how to lower the volume"

# Resultados de la busqueda en la base de datos vectorial
results = db.similarity_search_with_score(query, k=5)

# En caso de no encontrar ninguna coincidencia
if len(results) == 0:
    print("\n\n---\n\n",f"Unable to find matching results.")
    
# Crear el prompt final
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query)

# Carga del LLM
# llm = ChatOllama(model="llama3.2:3b")
llm = ChatOllama(model="llama3.1")

# Prompt-> LLM -> Recibe respuesta y imprime en consola 
ai_msg  = llm.invoke(prompt)
print(ai_msg.content)