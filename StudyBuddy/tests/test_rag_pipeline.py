import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from config.settings import settings

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        # Skip tests if no API key is provided
        if not os.getenv('GEMINI_API_KEY'):
            self.skipTest("GEMINI_API_KEY not set")
        
        self.rag = RAGPipeline()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test documents
        self.test_file1 = Path(self.test_dir) / "test1.txt"
        self.test_file1.write_text("This is a test document about machine learning. It covers neural networks and deep learning algorithms.")
        
        self.test_file2 = Path(self.test_dir) / "test2.txt"
        self.test_file2.write_text("Python is a programming language. It is widely used for data science and web development.")
    
    def test_ingest_documents(self):
        result = self.rag.ingest_documents(str(self.test_dir))
        
        self.assertTrue(result['success'])
        self.assertEqual(result['documents_processed'], 2)
        self.assertIsNotNone(result['stats'])
    
    def test_query_with_context(self):
        # First ingest documents
        self.rag.ingest_documents(str(self.test_dir))
        
        # Query about machine learning
        result = self.rag.query("What is machine learning?")
        
        self.assertTrue(result['success'])
        self.assertIsInstance(result['answer'], str)
        self.assertGreater(len(result['answer']), 0)
    
    def test_query_without_context(self):
        # Clear knowledge base first
        self.rag.clear_knowledge_base()
        
        result = self.rag.query("What is quantum computing?")
        
        self.assertTrue(result['success'])
        self.assertIn("couldn't find", result['answer'].lower())
    
    def test_add_single_document(self):
        result = self.rag.add_single_document(str(self.test_file1))
        
        self.assertTrue(result['success'])
        self.assertEqual(result['filename'], 'test1.txt')
    
    def test_knowledge_base_stats(self):
        self.rag.ingest_documents(str(self.test_dir))
        
        result = self.rag.get_knowledge_base_stats()
        
        self.assertTrue(result['success'])
        self.assertGreater(result['total_files'], 0)
        self.assertIsInstance(result['files'], list)
    
    def test_search_knowledge_base(self):
        self.rag.ingest_documents(str(self.test_dir))
        
        result = self.rag.search_knowledge_base("machine learning", top_k=3)
        
        self.assertTrue(result['success'])
        self.assertIsInstance(result['results'], list)
        self.assertGreater(len(result['results']), 0)
    
    def tearDown(self):
        # Clean up test files
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Clear knowledge base
        if hasattr(self, 'rag'):
            try:
                self.rag.clear_knowledge_base()
            except:
                pass

if __name__ == '__main__':
    unittest.main()