# backend/core/document_processor.py
import os
from typing import List, Dict, Tuple, Any
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import config
from backend.utils.logger import logger

class AdvancedDocumentProcessor:
    def __init__(self):
        # Multiple text splitters for different strategies
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        # Semantic chunker for better context preservation
        self.semantic_splitter = SemanticChunker(
            HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )
        
        # Markdown splitter for markdown files
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        logger.info("Initialized Advanced Document Processor")
    
    def load_document(self, filepath: str) -> List[Any]:
        """Load document based on file type"""
        ext = os.path.splitext(filepath)[1].lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.md': TextLoader
        }
        
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        
        try:
            loader_class = loaders[ext]
            
            # SPECIAL HANDLING FOR LARGE PDFS
            if ext == '.pdf':
                file_size = os.path.getsize(filepath)
                MAX_PDF_SIZE = 20 * 1024 * 1024  # 20MB
                
                if file_size > MAX_PDF_SIZE:
                    logger.warning(f"Large PDF detected ({file_size/(1024*1024):.1f}MB). Using optimized loading.")
                    # Use fast loading with page limit
                    return self._load_large_pdf_optimized(filepath)
            
            # Special handling for Excel to preserve tables
            if ext in ['.xlsx', '.xls']:
                loader = loader_class(filepath, mode="elements")
            else:
                loader = loader_class(filepath)
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {filepath}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load document {filepath}: {str(e)}")
            raise

    def _load_large_pdf_optimized(self, filepath: str) -> List[Any]:
        """Optimized loading for large PDFs"""
        try:
            from PyPDF2 import PdfReader
            from langchain_core.documents import Document
            
            reader = PdfReader(filepath)
            documents = []
            
            # Process only first 100 pages for large PDFs
            max_pages = min(100, len(reader.pages))
            logger.info(f"Processing {max_pages} of {len(reader.pages)} pages for large PDF")
            
            for page_num in range(max_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:  # Only add if meaningful text
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(filepath),
                                "page": page_num + 1,
                                "total_pages": len(reader.pages)
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            logger.info(f"Optimized loading: {len(documents)} pages extracted")
            return documents
            
        except Exception as e:
            logger.error(f"Optimized PDF loading failed: {str(e)}")
            # Fallback to regular loader
            loader = PyPDFLoader(filepath)
            return loader.load()

    def smart_chunking(self, documents: List[Any], filename: str) -> Tuple[List[str], List[Dict]]:
        """Apply intelligent chunking based on document type and content"""
        ext = os.path.splitext(filename)[1].lower()
        
        # Choose splitter based on file type
        if ext == '.md':
            # For markdown, use header-based splitting
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            chunk_docs = self.markdown_splitter.split_text(combined_text)
        elif ext in ['.pdf', '.docx'] and len(documents) > 10:
            # For large documents, use semantic chunking
            chunk_docs = self.semantic_splitter.split_documents(documents)
        else:
            # Default to recursive splitting
            chunk_docs = self.recursive_splitter.split_documents(documents)
        
        # Extract chunks and create metadata
        chunks = []
        metadata_list = []
        
        for i, chunk in enumerate(chunk_docs):
            chunks.append(chunk.page_content)
            
            # Enhanced metadata
            metadata = {
                "source": filename,
                "chunk_id": i,
                "total_chunks": len(chunk_docs),
                "file_type": ext[1:],
                "chunk_length": len(chunk.page_content),
                "page": chunk.metadata.get("page", 1) if hasattr(chunk, 'metadata') else 1,
                "has_table": self._contains_table(chunk.page_content),
                "contains_code": self._contains_code(chunk.page_content)
            }
            
            # Preserve original metadata
            if hasattr(chunk, 'metadata'):
                metadata.update(chunk.metadata)
            
            metadata_list.append(metadata)
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks, metadata_list
    
    def process_large_document_optimized(self, filepath: str, filename: str, max_pages: int = 50) -> Tuple[List[str], List[Dict]]:
        """Optimized processing for large documents"""
        try:
            from PyPDF2 import PdfReader
            from langchain_core.documents import Document
            
            reader = PdfReader(filepath)
            total_pages = len(reader.pages)
            
            # Limit pages for large documents
            if total_pages > max_pages:
                logger.warning(f"Large PDF ({total_pages} pages), processing first {max_pages} pages only")
                pages_to_process = max_pages
            else:
                pages_to_process = total_pages
            
            documents = []
            for page_num in range(pages_to_process):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "total_pages": total_pages
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            # Apply smart chunking
            chunks, metadata = self.smart_chunking(documents, filename)
            
            logger.info(f"Optimized processing: {len(chunks)} chunks from {pages_to_process}/{total_pages} pages")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Optimized processing failed: {str(e)}")
            raise

    def _contains_table(self, text: str) -> bool:
        """Check if text contains table-like structure"""
        table_indicators = ['|', '+---', '┌', '└', '├', '┼']
        return any(indicator in text for indicator in table_indicators)
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code"""
        code_indicators = ['def ', 'class ', 'import ', 'function ', '={', '};']
        return any(indicator in text for indicator in code_indicators)
    
    def process_document(self, filepath: str, filename: str) -> Tuple[List[str], List[Dict]]:
        """Main processing pipeline"""
        try:
            # Load document
            documents = self.load_document(filepath)
            
            # Apply smart chunking
            chunks, metadata = self.smart_chunking(documents, filename)
            
            # Validate chunks
            valid_chunks = []
            valid_metadata = []
            
            for chunk, meta in zip(chunks, metadata):
                if len(chunk.strip()) > 10:  # Minimum chunk size
                    valid_chunks.append(chunk.strip())
                    valid_metadata.append(meta)
            
            logger.info(f"Processed {filename}: {len(valid_chunks)} valid chunks")
            return valid_chunks, valid_metadata
            
        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {str(e)}")
            raise