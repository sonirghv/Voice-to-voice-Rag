"""
PDF Extraction Tool

Processes PDFs and stores embeddings, chunks, and metadata in a FAISS index.
Logs processing details to logger/parser.csv.
"""

import argparse
import os
import logging
import csv
import hashlib
import time
import gc
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Required imports
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import psutil
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import re
from pathlib import Path
import faiss

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Begin PDF extraction module code ---
@dataclass
class ExtractionResult:
    """Container for extraction results and metadata."""
    text: str
    page_count: int
    success: bool
    method_used: str
    document_id: str
    error: Optional[str] = None
    metadata: Dict = None

class PDFExtractor:
    """Handles PDF text extraction with multiple fallback methods."""
    
    def __init__(self, enable_ocr: bool = True, max_workers: int = 4):
        """Initialize the PDF extractor."""
        self.enable_ocr = enable_ocr
        self.max_workers = max_workers
        
        # Available extraction methods in order of preference
        self._extraction_methods = [
            self._extract_with_pymupdf,
            self._extract_with_pdfminer,
        ]
        
        if enable_ocr:
            self._extraction_methods.append(self._extract_with_ocr)
        
        logger.info(
            f"PDF-INIT: Initialized PDF extractor:\n"
            f"  • OCR enabled: {enable_ocr}\n"
            f"  • Max workers: {max_workers}\n"
            f"  • Available methods: {[m.__name__ for m in self._extraction_methods]}"
        )
    
    def _generate_document_id(self, pdf_path: str) -> str:
        """Generate a unique document ID based on file path and modification time."""
        stats = os.stat(pdf_path)
        mod_time = datetime.fromtimestamp(stats.st_mtime).isoformat()
        file_size = stats.st_size
        unique_string = f"{pdf_path}:{file_size}:{mod_time}"
        doc_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
        logger.debug(
            f"PDF-ID: Generated document ID:\n"
            f"  • Path: {pdf_path}\n"
            f"  • Size: {file_size/1024/1024:.1f}MB\n"
            f"  • Modified: {mod_time}\n"
            f"  • ID: {doc_id}"
        )
        return doc_id
    
    def extract_text(self, pdf_path: str) -> Tuple[bool, str, Dict]:
        """Extract text from PDF using available methods."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF-ERROR: File not found: {pdf_path}")
            return False, "", {}
        
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        
        logger.info(
            f"PDF-EXTRACT: Starting text extraction:\n"
            f"  • File: {pdf_path}\n"
            f"  • Size: {file_size:.1f}MB"
        )
        
        for method in self._extraction_methods:
            try:
                logger.info(f"PDF-EXTRACT: Trying {method.__name__}")
                success, text, metadata = method(pdf_path)
                
                if success and text.strip():
                    logger.info(
                        f"PDF-EXTRACT: Successfully extracted with {method.__name__}:\n"
                        f"  • Time: {metadata.get('time', 0):.2f}s\n"
                        f"  • Pages: {metadata.get('pages', -1)}\n"
                        f"  • Text length: {len(text)} chars"
                    )
                    return True, text, metadata
                
                logger.warning(
                    f"PDF-EXTRACT: {method.__name__} failed to extract text:\n"
                    f"  • Success: {success}\n"
                    f"  • Error: {metadata.get('error', 'No text extracted')}"
                )
                
            except Exception as e:
                logger.error(f"PDF-{method.__name__.upper()}-ERROR: Extraction failed: {str(e)}")
                continue
        
        logger.error(
            f"PDF-ERROR: All extraction methods failed for {pdf_path}:\n"
            f"  • Methods tried: {[m.__name__ for m in self._extraction_methods]}"
        )
        return False, "", {"error": "All extraction methods failed"}
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[bool, str, Dict]:
        """Extract text using PyMuPDF (MuPDF)."""
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 85:
            logger.warning(
                f"PDF-PYMUPDF-MEMORY-HIGH: Memory usage at {system_memory.percent:.1f}% before extraction. "
                f"Will use extra memory management."
            )
            gc.collect()
        
        try:
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            MAX_PAGES_PER_BATCH = 10 if system_memory.percent > 85 else 20
            
            def is_shutdown_requested():
                if hasattr(self, 'is_shutting_down') and self.is_shutting_down:
                    return True
                return False
            
            def extract_page_text(page_num):
                if is_shutdown_requested():
                    return None
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    page = None
                    return page_text
                except Exception as e:
                    logger.error(f"PDF-PYMUPDF-PAGE-ERROR: Error extracting page {page_num}: {str(e)}")
                    return ""
            
            all_text = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for batch_start in range(0, page_count, MAX_PAGES_PER_BATCH):
                    if is_shutdown_requested():
                        logger.warning(f"PDF-PYMUPDF-SHUTDOWN: Extraction interrupted at page {batch_start}/{page_count}")
                        doc.close()
                        return False, "".join(all_text), {"error": "Extraction interrupted by shutdown signal", "partial": True}
                    batch_end = min(batch_start + MAX_PAGES_PER_BATCH, page_count)
                    logger.info(f"PDF-PYMUPDF: Processing pages {batch_start+1}-{batch_end}/{page_count} with {self.max_workers} workers")
                    future_to_page = {executor.submit(extract_page_text, page_num): page_num for page_num in range(batch_start, batch_end)}
                    for future in future_to_page:
                        page_text = future.result()
                        if page_text is None:
                            logger.warning(f"PDF-PYMUPDF-SHUTDOWN: Extraction interrupted during parallel processing")
                            doc.close()
                            return False, "".join(all_text), {"error": "Extraction interrupted by shutdown signal", "partial": True}
                        all_text.append(page_text)
                    system_memory = psutil.virtual_memory()
                    if system_memory.percent > 90:
                        logger.warning(f"PDF-PYMUPDF-MEMORY-CRITICAL: Memory at {system_memory.percent:.1f}% during extraction")
                        gc.collect()
                        time.sleep(0.5)
            
            text = "".join(all_text)
            doc.close()
            gc.collect()
            memory_after = process.memory_info().rss
            memory_impact = (memory_after - memory_before) / (1024 * 1024)  # MB
            processing_time = time.time() - start_time
            
            if not text.strip():
                return False, "", {
                    "error": "No text extracted",
                    "time": processing_time,
                    "pages": page_count,
                    "memory": memory_impact
                }
            
            logger.info(
                f"PDF-PYMUPDF: Extraction complete:\n"
                f"  • Success: True\n"
                f"  • Time: {processing_time:.2f}s\n"
                f"  • Pages: {page_count}\n"
                f"  • Memory impact: {memory_impact:.1f}MB\n"
                f"  • Text length: {len(text)} chars"
            )
            
            return True, text, {
                "time": processing_time,
                "pages": page_count,
                "memory": memory_impact
            }
        except Exception as e:
            logger.error(f"PDF-PYMUPDF-ERROR: {str(e)}")
            return False, "", {"error": str(e)}
    
    def _extract_with_pdfminer(self, pdf_path: str) -> Tuple[bool, str, Dict]:
        """Extract text using PDFMiner."""
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 85:
            logger.warning(
                f"PDF-PDFMINER-MEMORY-HIGH: Memory usage at {system_memory.percent:.1f}% before extraction."
            )
            gc.collect()
        
        logger.info("PDF-PDFMINER: Starting extraction with PDFMiner")
        
        def is_shutdown_requested():
            if hasattr(self, 'is_shutting_down') and self.is_shutting_down:
                return True
            return False
        
        try:
            # For larger files, you can add page-by-page extraction using PDFMiner's low-level APIs.
            text = pdfminer_extract(pdf_path)
            gc.collect()
            memory_after = process.memory_info().rss
            memory_impact = (memory_after - memory_before) / (1024 * 1024)  # MB
            processing_time = time.time() - start_time
            
            if not text.strip():
                return False, "", {
                    "error": "No text extracted",
                    "time": processing_time,
                    "memory": memory_impact
                }
            
            logger.info(
                f"PDF-PDFMINER: Extraction complete:\n"
                f"  • Success: True\n"
                f"  • Time: {processing_time:.2f}s\n"
                f"  • Memory impact: {memory_impact:.1f}MB\n"
                f"  • Text length: {len(text)} chars"
            )
            
            return True, text, {
                "time": processing_time,
                "memory": memory_impact
            }
        except Exception as e:
            logger.error(f"PDF-PDFMINER-ERROR: {str(e)}")
            gc.collect()
            return False, "", {"error": str(e)}
    
    def _extract_with_ocr(self, pdf_path: str) -> Tuple[bool, str, Dict]:
        """Extract text using OCR (Tesseract) as last resort."""
        if not self.enable_ocr:
            return False, "", {"error": "OCR not enabled"}
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        system_memory = psutil.virtual_memory()
        
        if system_memory.percent > 90:
            logger.warning(
                f"PDF-OCR-MEMORY-CRITICAL: Memory usage at {system_memory.percent:.1f}% before OCR."
            )
            gc.collect()
            system_memory = psutil.virtual_memory()
            if system_memory.percent > 95:
                logger.warning(f"PDF-OCR-MEMORY-ADAPT: Memory too high ({system_memory.percent:.1f}%), using minimum quality settings")
                dpi = 100
                batch_size = 5
            else:
                dpi = 150
                batch_size = 10
        else:
            dpi = 300
            batch_size = 20
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            page_count = doc.page_count
            
            logger.info(
                f"PDF-OCR-START: Beginning OCR processing:\n"
                f"  • Total pages: {page_count}\n"
                f"  • DPI: {dpi}\n"
                f"  • Batch size: {batch_size}"
            )
            
            for page_num in range(page_count):
                if page_num % batch_size == 0:
                    system_memory = psutil.virtual_memory()
                    if system_memory.percent > 95:
                        logger.warning(f"PDF-OCR-MEMORY-CRITICAL: Memory at {system_memory.percent:.1f}% during OCR (page {page_num+1}/{page_count})")
                        gc.collect()
                        if dpi > 100:
                            old_dpi = dpi
                            dpi = 100
                            logger.warning(f"PDF-OCR-ADAPT: Reduced DPI from {old_dpi} to {dpi} due to memory pressure")
                        if batch_size > 5:
                            old_batch = batch_size
                            batch_size = 5
                            logger.warning(f"PDF-OCR-ADAPT: Reduced batch size from {old_batch} to {batch_size}")
                        time.sleep(0.5)
                try:
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                    text += page_text
                    img = None
                    pix = None
                    page = None
                    if page_num % batch_size == batch_size - 1:
                        gc.collect()
                    if (page_num + 1) % 50 == 0:
                        logger.info(f"PDF-OCR-PROGRESS: Processed {page_num + 1}/{page_count} pages")
                except Exception as e:
                    logger.error(f"PDF-OCR-PAGE-ERROR: Failed to process page {page_num}: {str(e)}")
                    continue
            
            doc.close()
            memory_after = process.memory_info().rss
            memory_impact = (memory_after - memory_before) / (1024 * 1024)
            processing_time = time.time() - start_time
            
            if not text.strip():
                return False, "", {
                    "error": "No text extracted with OCR",
                    "time": processing_time,
                    "pages": page_count,
                    "memory": memory_impact
                }
            
            logger.info(
                f"PDF-OCR: Extraction complete:\n"
                f"  • Success: True\n"
                f"  • Time: {processing_time:.2f}s\n"
                f"  • Pages: {page_count}\n"
                f"  • Memory impact: {memory_impact:.1f}MB\n"
                f"  • Text length: {len(text)} chars"
            )
            
            return True, text, {
                "time": processing_time,
                "pages": page_count,
                "memory": memory_impact,
                "method": "ocr"
            }
        except Exception as e:
            logger.error(f"PDF-OCR-ERROR: {str(e)}")
            return False, "", {"error": str(e)}
# --- End PDF extraction module code ---

# --- New Functionalities: Text Chunking, Embeddings, FAISS, and Logging ---
class TextChunker:
    """Handles text chunking with overlap."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if chunk_overlap >= chunk_size:
            logger.warning("Chunk overlap >= chunk size, setting overlap to chunk_size/2")
            self.chunk_overlap = chunk_size // 2
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_words = " ".join(current_chunk[-self.chunk_overlap:]).split()
                current_chunk = overlap_words[:self.chunk_overlap]
                current_length = len(current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class EmbeddingGenerator:
    """Generates embeddings using SentenceTransformer."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"EMBEDDING-INIT: Loaded SentenceTransformer model: {model_name}, dimension: {self.dimension}")
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        try:
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            logger.info(f"EMBEDDING: Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"EMBEDDING-ERROR: Failed to generate embeddings: {str(e)}")
            return np.array([])

class FAISSManager:
    """Manages FAISS index for storing embeddings, chunks, and metadata."""
    
    def __init__(self, dimension: int, index_path: str, chunk_info_path: str):
        """Initialize FAISS index with ID mapping."""
        self.dimension = dimension
        self.index_path = index_path
        self.chunk_info_path = chunk_info_path
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self.chunk_info = {}  # Maps chunk_id (int) to {'chunk': str, 'metadata': dict}
        self.current_id = 0
        logger.info(f"FAISS-INIT: Initialized FAISS IndexIDMap with dimension {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str], doc_id: str, doc_metadata: Dict) -> List[int]:
        """Add embeddings, chunks, and metadata to FAISS index."""
        if embeddings.size == 0 or not chunks:
            logger.warning("FAISS-ADD: No embeddings or chunks to add")
            return []
        
        chunk_ids = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            chunk_id = self.current_id
            self.chunk_info[chunk_id] = {
                'chunk': chunk,
                'metadata': {**doc_metadata, 'chunk_index': i}
            }
            chunk_ids.append(chunk_id)
            self.current_id += 1
        
        # Add embeddings to FAISS with IDs
        embeddings = embeddings.astype(np.float32)
        ids = np.array(chunk_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)
        logger.info(f"FAISS-ADD: Added {len(embeddings)} embeddings for doc_id {doc_id}")
        return chunk_ids
    
    def save(self):
        """Save FAISS index and chunk info to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.chunk_info_path, 'wb') as f:
                pickle.dump(self.chunk_info, f)
            logger.info(f"FAISS-SAVE: Saved index to {self.index_path} and chunk info to {self.chunk_info_path}")
        except Exception as e:
            logger.error(f"FAISS-SAVE-ERROR: Failed to save: {str(e)}")

def generate_keywords(text: str, filename: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text and filename."""
    words = re.findall(r'\b\w+\b', text.lower())
    filename_words = re.findall(r'\b\w+\b', filename.lower())
    stopwords = {'the', 'and', 'of', 'to', 'in', 'a', 'for', 'on', 'with', 'by'}
    words = [w for w in words if w not in stopwords and len(w) > 3]
    
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    for word in filename_words:
        if word in word_freq:
            word_freq[word] *= 2
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:max_keywords]]
    
    for word in filename_words:
        if word not in keywords and len(keywords) < max_keywords:
            keywords.append(word)
    
    return keywords[:max_keywords]

def generate_metadata(pdf_path: str, text: str, doc_id: str) -> Dict:
    """Generate metadata for a PDF file."""
    filename = os.path.basename(pdf_path)
    keywords = generate_keywords(text, filename)
    
    return {
        "filename": filename,
        "file_path": pdf_path,
        "file_extension": os.path.splitext(pdf_path)[1],
        "keywords": keywords,
        "document_id": doc_id,
        "last_modified": datetime.fromtimestamp(os.path.getmtime(pdf_path)).isoformat()
    }

def check_duplicate(doc_id: str, processed_ids: set) -> bool:
    """Check if document ID has been processed."""
    if doc_id in processed_ids:
        logger.warning(f"DUPLICATE: Document ID {doc_id} already processed")
        return True
    return False

def log_to_csv(record: Dict, csv_path: str):
    """Log processing record to CSV."""
    fieldnames = [
        'document_id', 'filename', 'file_path', 'success', 'page_count', 
        'text_length', 'chunk_count', 'embedding_count', 'processing_time', 
        'memory_impact', 'error'
    ]
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def main():
    # Hardcoded default configuration
    default_config = {
        "pdf_file": "./pdfs",  # Default folder for PDFs
        "output_dir": "output",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "max_workers": 4,
        "enable_ocr": True,
        "csv_log": "logger/parser.csv"
    }

    parser = argparse.ArgumentParser(description="PDF extraction tool with FAISS storage.")
    parser.add_argument("pdf_file", nargs='?', default=default_config["pdf_file"],
                        help="Path to PDF file or folder containing PDFs (default: ./pdfs)")
    parser.add_argument("--output_dir", default=default_config["output_dir"],
                        help="Output directory for FAISS index (default: output)")
    parser.add_argument("--chunk_size", type=int, default=default_config["chunk_size"],
                        help="Number of words per chunk (default: 500)")
    parser.add_argument("--chunk_overlap", type=int, default=default_config["chunk_overlap"],
                        help="Number of overlapping words between chunks (default: 50)")
    parser.add_argument("--max_workers", type=int, default=default_config["max_workers"],
                        help="Number of worker threads for extraction (default: 4)")
    parser.add_argument("--enable_ocr", action="store_true", default=default_config["enable_ocr"],
                        help="Enable OCR for scanned PDFs (default: True)")
    parser.add_argument("--csv_log", default=default_config["csv_log"],
                        help="Path to CSV log file (default: logger/parser.csv)")
    args = parser.parse_args()

    # Initialize components
    extractor = PDFExtractor(enable_ocr=args.enable_ocr, max_workers=args.max_workers)
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    embedder = EmbeddingGenerator()
    faiss_index_path = os.path.join(args.output_dir, "faiss_index.bin")
    faiss_chunk_info_path = os.path.join(args.output_dir, "faiss_chunk_info.pkl")
    faiss_manager = FAISSManager(dimension=embedder.dimension, index_path=faiss_index_path, chunk_info_path=faiss_chunk_info_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track processed document IDs
    processed_ids = set()
    
    # Check if input is a folder or single file
    if os.path.isdir(args.pdf_file):
        pdf_files = [os.path.join(args.pdf_file, f) for f in os.listdir(args.pdf_file) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files in {args.pdf_file}")
    else:
        pdf_files = [args.pdf_file]
    
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            logger.error(f"Input file does not exist: {pdf_path}")
            continue

        start_time = time.time()
        doc_id = extractor._generate_document_id(pdf_path)
        
        if check_duplicate(doc_id, processed_ids):
            continue
        
        processed_ids.add(doc_id)
        success, text, extract_metadata = extractor.extract_text(pdf_path)

        record = {
            'document_id': doc_id,
            'filename': os.path.basename(pdf_path),
            'file_path': pdf_path,
            'success': success,
            'page_count': extract_metadata.get('pages', 0),
            'text_length': len(text) if success else 0,
            'chunk_count': 0,
            'embedding_count': 0,
            'processing_time': 0,
            'memory_impact': extract_metadata.get('memory', 0),
            'error': extract_metadata.get('error', '') if not success else ''
        }

        if not success:
            logger.error(f"Extraction failed: {extract_metadata.get('error', 'Unknown error')}")
            log_to_csv(record, args.csv_log)
            continue

        # Generate metadata
        doc_metadata = generate_metadata(pdf_path, text, doc_id)
        
        # Chunk text
        chunks = chunker.chunk_text(text)
        record['chunk_count'] = len(chunks)
        
        # Generate embeddings
        embeddings = embedder.generate_embeddings(chunks)
        record['embedding_count'] = len(embeddings)
        
        # Add to FAISS index
        chunk_ids = faiss_manager.add_embeddings(embeddings, chunks, doc_id, doc_metadata)
        
        # Log processing time
        record['processing_time'] = time.time() - start_time
        log_to_csv(record, args.csv_log)
        
        logger.info(f"Processed {pdf_path}: {len(chunks)} chunks, {len(embeddings)} embeddings")
        
        # Clean up
        gc.collect()
    
    # Save FAISS index and chunk info
    faiss_manager.save()
    logger.info("Processing complete")

if __name__ == "__main__":
    main()