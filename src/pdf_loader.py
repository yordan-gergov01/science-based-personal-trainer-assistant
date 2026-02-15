"""PDF loading and processing with Course-specific categorization"""
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from pathlib import Path
from tqdm import tqdm
from .config import (
    PDF_DIRECTORY, CHUNK_SIZE, CHUNK_OVERLAP, 
    LOGS_PATH, MODULE_CATEGORIES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_PATH}/pdf_loader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_module_category(filename):
    """
    Extract module category from Course's PDF filenames
    
    Examples:
    - "Protein PTC 2022.pdf" -> "nutrition"
    - "Exercise Selection PTC 2022.pdf" -> "training"
    """
    filename_lower = filename.lower()
    
    # Remove common suffixes
    filename_lower = filename_lower.replace('ptc 2022', '').replace('ptc 2023', '')
    filename_lower = filename_lower.replace('.pdf', '').strip()
    
    # Check each category
    for category, keywords in MODULE_CATEGORIES.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return category
    
    return 'general'

def extract_topic_name(filename):
    """
    Extract clean topic name from filename
    
    Examples:
    - "Protein PTC 2022.pdf" -> "Protein"
    - "Exercise Selection PTC 2022 (1).pdf" -> "Exercise Selection"
    """
    # Remove PTC year markers
    name = filename.replace('PTC 2022', '').replace('PTC 2023', '')
    # Remove .pdf
    name = name.replace('.pdf', '')
    # Remove trailing numbers like (1)
    import re
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    # Clean up whitespace
    name = name.strip()
    
    return name

def load_pdfs(directory_path=PDF_DIRECTORY):
    """Load all PDFs with progress tracking"""
    logger.info(f"Loading PDFs from {directory_path}")
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found! Please add your PDFs to the 'pdfs' folder.")
    
    # Check if PDFs exist
    pdf_files = list(Path(directory_path).rglob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {directory_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Show file categorization
    print("\nPDF FILES BY CATEGORY:")
    category_counts = {}
    for pdf_file in pdf_files:
        category = extract_module_category(pdf_file.name)
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        print(f"  {category.upper()}: {count} files")
    print()
    
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    
    return documents

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into chunks with specific metadata"""
    logger.info(f"Splitting documents: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add Course-specific metadata
    print("\nAdding metadata to chunks...")
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_length'] = len(chunk.page_content)
        
        # Extract filename
        source = chunk.metadata.get('source', '')
        filename = os.path.basename(source)
        
        # Categorize
        chunk.metadata['category'] = extract_module_category(filename)
        chunk.metadata['topic'] = extract_topic_name(filename)
        chunk.metadata['course'] = 'Menno Henselmans PTC'
    
    logger.info(f"Created {len(chunks)} chunks")
    
    # Show chunk distribution
    category_chunks = {}
    for chunk in chunks:
        cat = chunk.metadata.get('category', 'unknown')
        category_chunks[cat] = category_chunks.get(cat, 0) + 1
    
    print("\nCHUNKS BY CATEGORY:")
    for category, count in sorted(category_chunks.items()):
        print(f"  {category.upper()}: {count} chunks")
    
    return chunks

def get_document_stats(documents):
    """Get statistics about loaded documents"""
    stats = {
        'total_pages': len(documents),
        'total_characters': sum(len(doc.page_content) for doc in documents),
        'avg_page_length': sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0,
        'unique_sources': len(set(doc.metadata.get('source', 'unknown') for doc in documents))
    }
    return stats

if __name__ == "__main__":
    print("Testing PDF Loader\n")
    
    # Load PDFs
    docs = load_pdfs()
    
    # Show stats
    stats = get_document_stats(docs)
    print("\nDOCUMENT STATISTICS:")
    print(f"  Total Pages: {stats['total_pages']}")
    print(f"  Total Characters: {stats['total_characters']:,}")
    print(f"  Avg Page Length: {stats['avg_page_length']:.0f} chars")
    print(f"  Unique PDFs: {stats['unique_sources']}")
    
    # Split
    chunks = split_documents(docs)
    
    # Show sample
    print("\nSAMPLE CHUNK:")
    print("="*80)
    sample = chunks[0]
    print(f"Topic: {sample.metadata.get('topic', 'Unknown')}")
    print(f"Category: {sample.metadata.get('category', 'Unknown')}")
    print(f"Length: {sample.metadata.get('chunk_length', 0)} chars")
    print(f"\nContent:\n{sample.page_content[:400]}...")
    print("="*80)