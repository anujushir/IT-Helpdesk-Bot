import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ================== STYLING ==================
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Header */
    .title {
        text-align: center;
        font-size: 36px;
        color: #00FFAA;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #B0B0B0;
        margin-bottom: 30px;
    }

    /* Chat Input */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 10px;
    }

    /* User Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Assistant Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Assistant Avatar */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }

    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ================== CONSTANTS ==================
PDF_PATH = r"GenAI-IT HelpdeskBot2.pdf"
PROMPT_TEMPLATE = """
You are a professional IT Helpdesk assistant. 
Answer user queries using the given context. 
Be clear, concise, and professional. 
If unsure, politely state that you don’t have the information.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

# ================== MODELS ==================
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# ================== FUNCTIONS ==================
def load_pdf_documents(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke(
        {"user_query": user_query, "document_context": context_text}
    )
    return response  # Clean — no "thinking" shown

# ================== MAIN APP ==================
st.markdown("<div class='title'>♠ GenAI-Powered IT Helpdesk Bot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>👨🏻‍💻 Welcome to Capgemini Intelligent IT Helpdesk</div>", unsafe_allow_html=True)
st.markdown("---")

# Load and process documents once
with st.spinner("Loading IT Helpdesk knowledge base..."):
    raw_docs = load_pdf_documents(PDF_PATH)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

st.success("✅ Knowledge base loaded! Ask me anything about IT Helpdesk policies and processes.")

# Chat UI
user_input = st.chat_input("Enter your IT Helpdesk question...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Fetching the best possible answer..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
