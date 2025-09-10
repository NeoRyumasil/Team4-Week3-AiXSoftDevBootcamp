# 🧠 Personal Knowledge Assistant

A powerful RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your personal notes, documents, and bookmarks using Google's Gemini AI.

## ✨ Features

- **📄 Multi-format Document Support**: Process PDF, DOCX, Markdown, TXT, and HTML files
- **🔍 Smart Retrieval**: Advanced semantic search with reranking
- **🤖 AI-Powered Responses**: Uses Gemini AI for natural language generation
- **💾 Vector Database**: ChromaDB for efficient similarity search
- **🎨 Clean Web Interface**: Streamlit-based UI with chat functionality
- **⚙️ Configurable**: Extensive configuration options via environment variables

## 🛠️ Tech Stack

### **Backend & AI**
- **Python 3.13** - Core programming language
- **Google Gemini AI** - Large language model for response generation
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **ChromaDB** - Vector database for similarity search
- **FastAPI** - Web framework for API endpoints
- **Pydantic** - Data validation and settings management

### **Document Processing**
- **PyPDF2** - PDF text extraction
- **python-docx** - Microsoft Word document processing
- **BeautifulSoup4** - HTML parsing and text extraction
- **Markdown** - Markdown file processing
- **tiktoken** - Token counting and text chunking

### **Frontend & Interface**
- **Streamlit** - Interactive web UI and chat interface
- **HTML/CSS** - Custom styling for chat components

### **Data & ML**
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning utilities
- **PyTorch** - Deep learning framework (via sentence-transformers)

### **Infrastructure**
- **ChromaDB** - Persistent vector storage
- **dotenv** - Environment variable management
- **Uvicorn** - ASGI server for FastAPI

### **Architecture Pattern**
- **RAG (Retrieval-Augmented Generation)** - Combines document retrieval with AI generation
- **Modular Design** - Separated ingestion, storage, retrieval, and generation components
- **Vector Embeddings** - Semantic similarity search using 384-dimensional vectors

## 🚀 Quick Start

### 1. Setup

```bash
# Clone or download the project
cd "Personal Knowledge Assistant"

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 2. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### 3. Run the Application

#### Option A: Web Interface (Recommended)
```bash
# Simple startup script (recommended)
python run.py

# Or using main.py
python main.py --ui

# Or directly with streamlit
streamlit run src/ui/streamlit_app.py
```

#### Option B: Command Line
```bash
# Ingest documents
python main.py --ingest ./data/documents

# Ask questions
python main.py --query "What is my project about?"

# View stats
python main.py --stats
```

## 📁 Project Structure

```
Personal Knowledge Assistant/
├── src/
│   ├── ingestion/          # Document processing
│   ├── database/           # Vector database management
│   ├── retrieval/          # Information retrieval
│   ├── generation/         # LLM integration
│   ├── ui/                 # Streamlit web interface
│   └── utils/              # Utility functions
├── config/                 # Configuration settings
├── data/
│   ├── documents/          # Your source documents
│   └── embeddings/         # Vector database storage
├── tests/                  # Test files
├── main.py                 # CLI entry point
└── requirements.txt        # Dependencies
```

## 📖 Usage Guide

### Adding Documents

1. **Via Web Interface**: Use the file uploader in the sidebar
2. **Manual**: Place files in `./data/documents/` and click "Process Documents"
3. **CLI**: Use `python main.py --ingest /path/to/documents`

### Supported File Types

- **Text**: `.txt`, `.md` 
- **Documents**: `.pdf`, `.docx`
- **Web**: `.html`

### Chat Interface

- Ask questions about your documents
- View sources for each response
- Conversation history is maintained
- Clear chat anytime with the sidebar button

### Configuration

Edit `.env` or modify `config/settings.py`:

```env
# Model Settings
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=all-MiniLM-L6-v2
TEMPERATURE=0.7
MAX_TOKENS=8192

# Retrieval Settings
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🛠️ Advanced Usage

### Custom Document Processing

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Add single document
result = rag.add_single_document("./my_document.pdf")

# Remove document
result = rag.remove_document("my_document.pdf")

# Search without generating response
results = rag.search_knowledge_base("search query", top_k=10)
```

### API Integration

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Query with conversation history
response = rag.query(
    question="What are the key findings?",
    conversation_history=[
        {"user": "Tell me about the project", "assistant": "This project is about..."}
    ]
)

print(response['answer'])
print(response['sources'])
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Gemini API key is correctly set in `.env`
2. **No Documents Found**: Check that files are in `./data/documents/` and are supported formats
3. **ChromaDB Telemetry Warnings**: These are harmless warnings that have been suppressed in the latest version
4. **Memory Issues**: Reduce `CHUNK_SIZE` or `TOP_K_RESULTS` in settings
5. **Slow Performance**: Use a smaller embedding model or reduce document size
6. **Torch Class Warnings**: These are non-critical warnings from the ML libraries and can be ignored

### Dependencies

If you encounter import errors:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📊 Performance Tips

- **Chunk Size**: Smaller chunks (500-1000) work better for specific questions
- **Embedding Model**: `all-MiniLM-L6-v2` is fast, `all-mpnet-base-v2` is more accurate
- **Document Quality**: Clean, well-structured documents yield better results
- **Query Specificity**: More specific questions get better answers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration options
3. Open an issue with detailed error messages

---

Happy chatting with your knowledge base! 🚀