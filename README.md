# 🌟 Genesis: RAG + LLM

Este proyecto combina **RAG (Retrieval-Augmented Generation)** y **LLM (Large Language Model)** para crear un sistema que puede buscar información relevante y generar respuestas precisas utilizando modelos avanzados. Se ha utilizado el modelo **mxbai-embed-large** para generar los embeddings, que luego se almacenan en la base de datos vectorial **ChromaDB**.

🔍🦙 **LLM Usado**: **Llama 3.2**  
⚡**Embeddings Model**: **mxbai-embed-large**  
💾 **Base de Datos Vectorial**: [ChromaDB](https://www.trychroma.com) | [GitHub: ChromaDB](https://github.com/chroma-core/chroma)  
🦙 **Gestión de Modelos**: [Ollama](https://ollama.com/)  
🦜️🔗 [Langchain]( https://www.langchain.com/)  
🎨**GUI [Streamlit](https://streamlit.io/)**
## ⚙️ Instalación
Pasos para instalar

### 1. Clonar el repositorio
```bash
git clone https://github.com/jfco09/genesis
```

### 2. Prerrequisitos
Instalar: 
1. 🐍 Python https://python.org/
2. 🦙 Ollama https://ollama.com/


### 3. Crear entorno virtual e instalar dependencias
Recomendable el uso de entorno virtual  
Creación de entorno virtual:

```bash
# Crear entorno virtual
python -m venv venv
```
Activar entorno virtual:
```bash
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```
Instalar dependencias:

```bash
pip install -r requirements.txt
```

### 4. Instalar los modelos en Ollama
Para instalar los modelos pon lo siguiente en una terminal:
1. 🦙Llama 3.2
```bash
ollama pull llama3.2
```
2. ⚡Instalar el modelo de embeddings mxbai-embed-large:

```bash
ollama pull mxbai-embed-large
```


## 🏃‍♂️ Ejecutar el proyecto
1. Crear la base de datos vectorial
Coloca los documentos en la carpeta ```data/``` y luego ejecuta el siguiente script para crear la base de datos de embeddings:

```bash
python create_database.py
```

2. Opciones de ejecución

🎨 Con interfaz gráfica:
```bash
streamlit run streamlit_ui.py
```

🖥️Usando script:
Edita el script ```query.py``` y modifica la siguiente línea:
```bash
response = retrieval_chain.invoke("PREGUNTA AQUI")
```
Ejecuta el script
```bash
python query.py
```
