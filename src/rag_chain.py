from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from .vector_store import load_vector_store
from .config import RETRIEVAL_K, SEARCH_TYPE, TEMPERATURE, LOGS_PATH
import os
import logging
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_PATH}/rag_chain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """You are an expert fitness coach trained in Menno Henselmans' evidence-based methodology. Answer client questions using your training expertise.

INSTRUCTIONS:
- Speak directly as a coach: "I recommend...", "You should...", "Based on the science..."
- Provide specific, actionable advice with numbers and protocols
- Base ALL answers strictly on the CONTEXT below (your training knowledge)
- Never mention "the course", "materials", or "according to..." - this is YOUR expertise
- If info isn't in context, say: "That's outside my specific area of expertise"
- Be confident but precise

YOUR TRAINING KNOWLEDGE:
{context}

CLIENT: {question}

YOUR ANSWER (as an expert coach):"""

def create_rag_chain(
    temperature=TEMPERATURE, 
    retrieval_k=RETRIEVAL_K,
    verbose=False
):
    """Create RAG chain with Groq"""
    logger.info(f"Creating RAG chain with Groq")
    logger.info(f"  Model: {GROQ_MODEL}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Retrieval K: {retrieval_k}")
    logger.info(f"  Cost per query: $0.00 (FREE)")
    
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env file!")
    
    # Load vector store
    vectorstore = load_vector_store()
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": retrieval_k}
    )
    
    # Create prompt
    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # Create Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
        max_tokens=1000
    )
    
    # Create chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    logger.info("✅ Groq RAG chain ready")
    return qa_chain

def ask_question(chain, question, verbose=True):
    """Ask question with Groq chain"""
    logger.info(f"Question: {question}")
    
    start_time = time.time()
    
    try:
        result = chain.invoke({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]
        elapsed_time = time.time() - start_time
        
        if verbose:
            # Group sources by topic
            topics = {}
            for doc in sources:
                topic = doc.metadata.get('topic', 'Unknown')
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(doc)
        
        # Log query
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer_length": len(answer),
            "num_sources": len(sources),
            "elapsed_time": elapsed_time,
            "topics_used": list(set(doc.metadata.get('topic', 'Unknown') for doc in sources)),
            "model": GROQ_MODEL,
            "success": True
        }
        
        with open(f"{LOGS_PATH}/queries.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return answer, sources
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        
        # Log error
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "error": str(e),
            "model": GROQ_MODEL,
            "success": False
        }
        
        with open(f"{LOGS_PATH}/queries.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return None, []

if __name__ == "__main__":
    
    # Create chain
    print("Loading chain...")
    chain = create_rag_chain()
    print("✅ Ready!\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Goodbye!")
            break
        
        try:
            answer, sources = ask_question(chain, question)
        except Exception as e:
            print(f"❌ Error: {e}")