import streamlit as st
import logging

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma  import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever


# Configuración streamlit
st.set_page_config(
    page_title="Proyecto Genesis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def process_question(question: str, db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"""Processing question: {question}""")
    llm = ChatOllama(model=selected_model, temperature=0)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate in english 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # retriever = MultiQueryRetriever.from_llm(
    #     db.as_retriever(kwargs={"k": 3}), llm, prompt=QUERY_PROMPT
    # )
    retriever = db.as_retriever(kwargs={"k": 5})

    PROMPT_TEMPLATE = """
    You are a helpful assistant that answers the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    Answer the question in SPANISH!!
    """

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("Demo UI", divider="gray", anchor=False)
    
    db_path = "db"



    try:
        # Carga del modelo para los embeddings
        embedding_function = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)

        # Carga de la base de datos
        db = Chroma(embedding_function=embedding_function ,persist_directory= db_path, collection_name = "rag-test")
        
    except:
        logger.info("Error al cargar base de datos")



    if "messages" not in st.session_state:
        st.session_state["messages"] = []



    
    message_container = st.container( border=True)

    for message in st.session_state["messages"]:
        avatar = "🤖" if message["role"] == "assistant" else "🥸"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="🥸").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    
                    response = process_question(
                        question=prompt, db=db, selected_model="llama3.2:3b"
                    )
                    st.markdown(response)
                    

            
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )

        except Exception as e:
            st.error(e, icon="⛔️")
            logger.error(f"Error processing prompt: {e}")


if __name__ == "__main__":
    main()