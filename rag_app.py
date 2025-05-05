import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
import tempfile

# ========== App Configuration ==========
st.set_page_config(
    page_title="RAG APP",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== Sidebar ==========
with st.sidebar:
    st.title("🔑 AI Configuration")
    
    # API Key Inputs
    groq_api_key = st.text_input("Groq API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")
    
    model_name = st.selectbox(
        "Groq Model",
        ["Llama3-8b-8192", "Llama3-70b-8192", "Mixtral-8x7b-32768"],
        index=0
    )

    temperature = st.slider("Creativeness", 0.0, 1.0, 0.2)
    memory_length = st.slider("Memory (messages)", 1, 10, 3)

    st.markdown("---")
    st.title("📄 Document Setup")
    st.markdown("""<div class="warning-box"><strong>Note:</strong> Answers are based only on document content.</div>""", unsafe_allow_html=True)

# Check for API keys
if not groq_api_key:
    st.error("Please enter your Groq API Key in the sidebar.")
    st.stop()

if not google_api_key:
    st.error("Please enter your Google API Key in the sidebar.")
    st.stop()

# ========== Core Functionality ==========
def initialize_llm():
    """Initialize Groq LLM with user-provided API key"""
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature
    )

llm = initialize_llm()

# Initialize embeddings with user-provided API key
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# ========== Document Processing ==========
def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        final_documents = text_splitter.split_documents(docs[:20])
        vectors = FAISS.from_documents(final_documents, embeddings)

        os.unlink(tmp_path)
        return vectors, len(final_documents), os.path.basename(uploaded_file.name)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, 0, ""

# ========== Conversation Chain ==========
def get_conversation_chain(vectorstore):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Return the original question if 
        it's not related to the document content."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(),
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document assistant. Answer questions 
        based only on the below context and chat history. Be precise and factual. 
        If you don't know the answer, say you don't know.

        Context: {context}

        Chat history: {chat_history}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ========== File Upload ==========
uploaded_file = st.file_uploader("Upload PDF (max 20 pages)", type="pdf")

if uploaded_file and ("vectors" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name):
    with st.status("Processing document...", expanded=True) as status:
        vectors, doc_count, filename = process_pdf(uploaded_file)
        if vectors:
            st.session_state.vectors = vectors
            st.session_state.current_file = uploaded_file.name
            status.update(label=f" Ready! Processed {doc_count} sections from {filename}", state="complete")
            st.balloons()
        else:
            status.update(label=" Processing failed", state="error")

# ========== Chat Interface ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.messages[-memory_length*2:]:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if user_input := st.chat_input("Ask about the document..."):
    if "vectors" not in st.session_state:
        st.warning("Please upload a PDF first")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing..."):
        try:
            conversation_chain = get_conversation_chain(st.session_state.vectors)
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            response = conversation_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")