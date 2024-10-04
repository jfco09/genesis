import os, shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from langchain_chroma  import Chroma

# Paths
dir_path = "data\\manuals"
db_path = "db"

# Manejo de la carga de los documentos
def load_documents():
    
    # Carga documentos del path y guarda en documents
    loader = DirectoryLoader(dir_path)
    documents = loader.load()
    
    return documents

# Divide los documentos en chunks "trozos"
def split_tex(documents: list[Document]):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 300,
        length_function = len,
        add_start_index = True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    test_chunk = chunks[1]
    
    print(test_chunk.page_content)
    print(test_chunk.metadata)
    
    return chunks
   
# Creacion de la base de datos vectorial
def create_chromadb(chunks: list[Document]):
    
    # Si existe -> elimina
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    # Crea la base de datos  usando el modelo mxbai-embed-large

    db = Chroma.from_documents(
        chunks, 
        OllamaEmbeddings(model="mxbai-embed-large", show_progress=True),
        persist_directory= db_path,
        collection_name = "rag-test",
    )
    
    print(f"Guardados {len(chunks)} chunks en {db_path}")
    

def main():
    
    documents = load_documents()
    
    chunks = split_tex(documents)
    
    create_chromadb(chunks)
    
if __name__ == "__main__":
    main()
