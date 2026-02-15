from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .pdf_loader import load_pdfs, split_documents
from .config import VECTOR_DB_PATH, EMBEDDING_MODEL, LOGS_PATH
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_PATH}/vector_store.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_vector_store(chunks=None, persist_directory=VECTOR_DB_PATH, force_recreate=False):
    """Create FAISS vector store"""
    
    os.makedirs(persist_directory, exist_ok=True)

    index_file = os.path.join(persist_directory, "index.faiss")
    if os.path.exists(index_file) and not force_recreate:
        logger.warning(f"FAISS index exists at {persist_directory}")
        response = input("Recreate it? (y/n): ")
        if response.lower() != 'y':
            logger.info("Loading existing index")
            return load_vector_store(persist_directory)
    
    if chunks is None:
        logger.info("Loading and processing PDFs...")
        docs = load_pdfs()
        chunks = split_documents(docs)
    
    logger.info(f"Creating FAISS vector store with {len(chunks)} chunks")
    logger.info("Cost: $0.00 (100% local)")
    
    print("\nInitializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("\nCreating FAISS index (this takes ~5-10 minutes)...")
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Save to disk
    print(f"\nSaving index to: {persist_directory}")
    vectorstore.save_local(persist_directory)
    
    logger.info(f"FAISS index created with {len(chunks)} embeddings")
    logger.info(f"Total cost: $0.00")
    logger.info(f"Saved to: {persist_directory}")
    
    return vectorstore

def load_vector_store(persist_directory=VECTOR_DB_PATH):
    """Load existing FAISS index"""
    logger.info(f"Loading FAISS index from {persist_directory}")
    
    index_file = os.path.join(persist_directory, "index.faiss")
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at {persist_directory}\n"
            f"Run: python -m src.vector_store"
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    count = vectorstore.index.ntotal
    logger.info(f"Loaded {count} embeddings from FAISS index")
    
    return vectorstore

def test_retrieval(vectorstore, query, k=5):
    """Test retrieval quality"""
    logger.info(f"Testing query: '{query}'")
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (similarity: {1/(1+score):.3f})")
        print(f"   Topic: {doc.metadata.get('topic', 'Unknown')}")
        print(f"   Category: {doc.metadata.get('category', 'Unknown')}")
        print(f"   Content: {doc.page_content[:200]}...")
        print(f"   {'-'*76}\n")
    
    return results

def get_vectorstore_stats(vectorstore):
    """Get statistics about FAISS index"""
    stats = {
        'total_embeddings': vectorstore.index.ntotal,
        'embedding_dimension': vectorstore.index.d,
    }
    return stats

if __name__ == "__main__":
    # Create vector store
    vectorstore = create_vector_store(force_recreate=True)
    
    # Show stats
    stats = get_vectorstore_stats(vectorstore)
    print("\nFAISS INDEX STATS:")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")