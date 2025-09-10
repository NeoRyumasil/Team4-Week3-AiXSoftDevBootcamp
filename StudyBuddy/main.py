#!/usr/bin/env python3
"""
Personal Knowledge Assistant - Main Entry Point

Run with: python main.py
Or use the Streamlit UI: streamlit run src/ui/streamlit_app.py
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

# Set environment variables before importing anything else
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*telemetry.*")

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.rag_pipeline import RAGPipeline
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description='Personal Knowledge Assistant')
    parser.add_argument('--ui', action='store_true', help='Launch Streamlit UI')
    parser.add_argument('--ingest', type=str, help='Ingest documents from directory')
    parser.add_argument('--query', type=str, help='Ask a question')
    parser.add_argument('--clear', action='store_true', help='Clear knowledge base')
    parser.add_argument('--stats', action='store_true', help='Show knowledge base stats')
    
    args = parser.parse_args()
    
    # Check for required API key
    try:
        settings.validate_required_keys()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Please copy .env.example to .env and add your Gemini API key")
        return
    
    if args.ui:
        import subprocess
        subprocess.run([
            'streamlit', 'run', 'src/ui/streamlit_app.py',
            '--server.headless=false',
            '--server.runOnSave=false',
            '--browser.gatherUsageStats=false'
        ])
        return
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return
    
    if args.ingest:
        print(f"📂 Ingesting documents from: {args.ingest}")
        result = rag.ingest_documents(args.ingest)
        
        if result['success']:
            print(f"✅ Successfully processed {result['documents_processed']} documents")
            print(f"📊 Total chunks in knowledge base: {result['stats']['total_chunks']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    elif args.query:
        print(f"🤔 Question: {args.query}")
        print("🔍 Searching knowledge base...")
        
        result = rag.query(args.query)
        
        if result['success']:
            print(f"\n💡 Answer: {result['answer']}")
            if result.get('sources'):
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
        else:
            print(f"❌ Error: {result['error']}")
    
    elif args.clear:
        print("🗑️ Clearing knowledge base...")
        result = rag.clear_knowledge_base()
        
        if result['success']:
            print("✅ Knowledge base cleared successfully")
        else:
            print(f"❌ Error: {result['error']}")
    
    elif args.stats:
        print("📊 Knowledge Base Statistics:")
        result = rag.get_knowledge_base_stats()
        
        if result['success']:
            stats = result['stats']
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Collection: {stats.get('collection_name', 'N/A')}")
            print(f"   Embedding model: {stats.get('embedding_model', 'N/A')}")
            print(f"   Total files: {result['total_files']}")
            
            if result['files']:
                print("\n📄 Indexed files:")
                for file in result['files'][:10]:
                    print(f"   • {file}")
                if len(result['files']) > 10:
                    print(f"   ... and {len(result['files']) - 10} more")
        else:
            print(f"❌ Error: {result['error']}")
    
    else:
        print("🧠 Personal Knowledge Assistant")
        print("\nUsage:")
        print("  python main.py --ui                    # Launch web interface")
        print("  python main.py --ingest ./documents    # Ingest documents")
        print("  python main.py --query 'your question' # Ask a question")
        print("  python main.py --stats                 # Show stats")
        print("  python main.py --clear                 # Clear knowledge base")
        print("\n💡 For the best experience, use: python main.py --ui")

if __name__ == "__main__":
    main()