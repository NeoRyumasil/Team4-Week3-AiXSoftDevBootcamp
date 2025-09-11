import streamlit as st
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug info (remove in production)
# print(f"Project root: {project_root}")
# print(f"Python path: {sys.path[:3]}...")  # Print first 3 items

try:
    from src.ingestion.document_processor import DocumentProcessor
    from src.database.vector_store import VectorStore
    from src.retrieval.retriever import Retriever
    from src.generation.llm_client import GeminiClient
    from config.settings import settings
    # print("✅ All imports successful!")  # Remove debug print
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(f"Current working directory: {os.getcwd()}")
    st.error(f"Python path: {sys.path}")
    raise

# Page configuration
st.set_page_config(
    page_title="StudyBuddy - Smart Study Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container padding */
    .main > div {
        padding-top: 2rem;
    }

    /* Chat message container */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
    }

    /* User message — biru muda segar */
    .stChatMessage[data-testid="chat-message"] > div:first-child > div:first-child[data-testid="user"] {
        background-color: #e3f2fd !important;
        border-left: 4px solid #1976d2;
        border-top-left-radius: 12px !important;
        border-top-right-radius: 12px !important;
    }

    /* Assistant message — biru ungu lembut */
    .stChatMessage[data-testid="chat-message"] > div:first-child > div:first-child[data-testid="assistant"] {
        background-color: #f0f7ff !important;
        border-left: 4px solid #3f51b5;
        border-top-left-radius: 12px !important;
        border-top-right-radius: 12px !important;
    }

    /* Hover effect for messages */
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    /* Source info styling */
    .source-info {
        font-size: 0.85rem;
        color: #1565c0;
        font-style: italic;
        margin-top: 0.75rem;
        padding: 0.5rem;
        background-color: #e8f4fd;
        border-radius: 8px;
        border-left: 3px solid #0d47a1;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #c0d9fc !important;
        color: #247fff;
        border: 3px solid #247fff
    }

    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton button {
        background-color: #ffffff20;
        color: #033985;
        border: 1px solid #c461cf;
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #ffffff30;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255,255,255,0.2);
    }

    /* Header styling */
    h1, h2, h3 {
        color: #1565c0;
        font-weight: 600;
    }

    /* Divider styling */
    hr {
        border-top: 2px solid #1976d2;
        margin: 1.5rem 0;
    }

    /* Success/info message styling */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
        border-left: 4px solid #1976d2;
    }

    /* Chat input styling */
    .stChatInput {
        padding-top: 1rem;
    }

    /* Spinner text */
    .stSpinner {
        color: #1976d2;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #0d47a1 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #1976d2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def initialize_components():
    """Initialize RAG components"""
    try:
        if st.session_state.vector_store is None:
            with st.spinner("Initializing vector store..."):
                st.session_state.vector_store = VectorStore()
        
        if st.session_state.retriever is None:
            with st.spinner("Initializing retriever..."):
                st.session_state.retriever = Retriever()
        
        if st.session_state.llm_client is None:
            with st.spinner("Initializing Gemini client..."):
                st.session_state.llm_client = GeminiClient()
        
        return True
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return False

def process_documents():
    doc_processor = DocumentProcessor()
    documents_path = Path(settings.DOCUMENTS_DIRECTORY)
    metadata_file = documents_path / ".processed_metadata.json"
    
    if not documents_path.exists():
        documents_path.mkdir(parents=True, exist_ok=True)
        st.info(f"Created documents directory at: {documents_path}")
        return
    
    with st.spinner("Processing documents..."):
        # Hanya proses file baru/berubah
        documents = doc_processor.process_directory(documents_path, metadata_file)
        
        if not documents:
            st.info("No new or modified documents to process.")
            return
        
        # Clear & re-add — atau lebih baik: update by source (lihat catatan di bawah)
        st.session_state.vector_store.clear_collection()
        st.session_state.vector_store.add_documents(documents)
        
        st.success(f"✅ Successfully processed and indexed {len(documents)} documents!")

def main():
    st.title("🧠 Smart Study Assistant")
    st.markdown("Chat with us to help improve your study")
    
    # Check for API key
    if not settings.GEMINI_API_KEY:
        st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        st.info("1. Copy .env.example to .env\n2. Add your Gemini API key\n3. Restart the application")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # # Document processing
        # if st.button("📤 Process Documents", use_container_width=True):
        #     if initialize_components():
        #         process_documents()
        
        # Upload files
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf', 'docx']
        )
        
        if uploaded_files:
            documents_path = Path(settings.DOCUMENTS_DIRECTORY)
            documents_path.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = documents_path / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"Uploaded {len(uploaded_files)} files!")

            if initialize_components():
                process_documents()
        
        st.divider()
        
        # Knowledge base stats
        st.header("📊 Knowledge Base Stats")
        if initialize_components():
            try:
                stats = st.session_state.retriever.get_stats()
                st.metric("Total Chunks", stats.get('total_chunks', 0))
                st.metric("Collection", stats.get('collection_name', 'N/A'))
                
                # List files in knowledge base
                files = st.session_state.vector_store.list_files()
                if files:
                    st.subheader("Indexed Files")
                    for file in files[:10]:  # Show first 10 files
                        st.text(f"📄 {file}")
                    if len(files) > 10:
                        st.text(f"... and {len(files) - 10} more")
                else:
                    st.info("No files indexed yet. Upload and process documents to get started!")
            except Exception as e:
                st.warning(f"Could not load stats: {str(e)}")
                st.info("Try processing some documents first.")
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize components
    if not initialize_components():
        return
    
    # Main chat interface
    st.header("💬directly chat to AI or 💾Insert the document you had ")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.markdown(f'<div class="source-info">Sources: {", ".join(message["sources"])}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating response..."):
                # Retrieve relevant context
                context_data = st.session_state.retriever.get_context_for_query(prompt)
                context = context_data['context']
                sources = context_data['sources']
                
                if not context.strip():
                    response = "I couldn't find any relevant information in your knowledge base to answer this question. Please make sure you have uploaded and processed relevant documents."
                    sources = []
                else:
                    # Generate response using LLM
                    conversation_history = [
                        {"user": msg["content"], "assistant": st.session_state.messages[i+1]["content"]} 
                        for i, msg in enumerate(st.session_state.messages[:-1]) 
                        if msg["role"] == "user" and i+1 < len(st.session_state.messages)
                    ]
                    
                    result = st.session_state.llm_client.generate_response(
                        prompt, context, conversation_history
                    )
                    response = result['response']
                
                # Display response
                st.markdown(response)
                
                if sources:
                    st.markdown(f'<div class="source-info">Sources: {", ".join(sources)}</div>', 
                               unsafe_allow_html=True)
        
        # Add assistant message to chat history
        assistant_message = {"role": "assistant", "content": response}
        if sources:
            assistant_message["sources"] = sources
        st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()