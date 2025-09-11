import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
import markdown
from bs4 import BeautifulSoup

import src.utils.text_processing as text_utils

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {'.txt', '.md', '.pdf', '.docx', '.html'}
    
    def process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_formats:
            print(f"Unsupported file format: {file_extension}")
            return None
        
        try:
            text_content = self._extract_text(file_path)
            if not text_content:
                return None
            
            cleaned_text = text_utils.clean_text(text_content)
            metadata = text_utils.extract_metadata_from_text(cleaned_text, file_path.name)
            
            return {
                'content': cleaned_text,
                'metadata': {
                    **metadata,
                    'file_path': str(file_path),
                    'file_extension': file_extension,
                    'file_size': file_path.stat().st_size,
                    'last_modified': file_path.stat().st_mtime
                }
            }
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        return {
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime,
            'file_path': str(file_path.resolve())  # absolute path
        }

    def _load_processed_metadata(self, metadata_file: Path) -> Dict[str, Dict]:
        """Muat metadata file yang sudah diproses sebelumnya."""
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def _save_processed_metadata(self, metadata_file: Path, metadata: Dict[str, Dict]):
        """Simpan metadata file yang baru saja diproses."""
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _extract_text(self, file_path: Path) -> str:
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self._extract_from_txt(file_path)
        elif file_extension == '.md':
            return self._extract_from_markdown(file_path)
        elif file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension == '.html':
            return self._extract_from_html(file_path)
        else:
            return ""
    
    def _extract_from_txt(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    def _extract_from_markdown(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            md_content = file.read()
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_html(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
    
    def process_directory(
        self, 
        directory_path: Path, 
        metadata_file: Optional[Path] = None,
        force_reprocess: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Proses file dalam direktori — hanya yang baru atau berubah.
        
        Args:
            directory_path: Path direktori dokumen
            metadata_file: Path file metadata (opsional, default: .processed_metadata.json di direktori)
            force_reprocess: Jika True, proses semua file tanpa cek metadata
        """
        documents = []
        
        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return documents
        
        # Default metadata file
        if metadata_file is None:
            metadata_file = directory_path / ".processed_metadata.json"
        
        # Muat metadata lama
        if force_reprocess:
            processed_metadata = {}
        else:
            processed_metadata = self._load_processed_metadata(metadata_file)
        
        current_metadata = {}
        files_to_process = []
        
        # Kumpulkan semua file yang valid
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                if file_path.suffix.lower() in self.supported_formats:
                    current_meta = self._get_file_metadata(file_path)
                    current_metadata[file_path.name] = current_meta
                    
                    # Cek apakah file baru atau berubah
                    if (force_reprocess or 
                        file_path.name not in processed_metadata or 
                        processed_metadata[file_path.name] != current_meta):
                        files_to_process.append(file_path)
                    else:
                        print(f"Skipping unchanged file: {file_path.name}")
                else:
                    print(f"Skipping unsupported file: {file_path.name}")
        
        # Proses hanya file yang perlu
        for file_path in files_to_process:
            processed_doc = self.process_file(file_path)
            if processed_doc:
                documents.append(processed_doc)
        
        # Simpan metadata baru setelah proses selesai
        if files_to_process:
            # Update metadata: gabung yang lama + yang baru diproses
            for fname in current_metadata:
                processed_metadata[fname] = current_metadata[fname]
            self._save_processed_metadata(metadata_file, processed_metadata)
            print(f"✅ Processed {len(files_to_process)} files. Metadata updated.")
        else:
            print("✅ No new or modified files to process.")
        
        return documents
    
    def save_processed_documents(self, documents: List[Dict[str, Any]], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, default=str)