# Evidence-Based Fitness Coach RAG System

AI-powered fitness coaching system trained on evidence-based methodology. Delivers accurate, science-backed answers through Retrieval-Augmented Generation.

## Key Features

- **RAG Architecture**: Source-grounded answers with citation support
- **Fast Retrieval**: FAISS vector search with semantic embeddings
- **Powerful LLM**: Llama 3.3 70B via Groq API (sub-2s responses)
- **100% Free**: Zero API costs, fully open-source stack

## Live Demo

**[Try it now](https://science-based-personal-trainer-assistant-e9efpqoqmhphugfwvhx8g.streamlit.app)** 

Ask questions about training, nutrition, recovery, and exercise science.

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Precision@6** | 88.3% |
| **Category Accuracy** | 85.0% |
| **Mean Reciprocal Rank** | 0.950 |
| **Average Latency** | 71ms |

> 📈 For detailed evaluation methodology and results, see `notebooks/evaluation.ipynb`

## Tech Stack
```
LangChain + FAISS + Sentence Transformers + Groq (Llama 3.3 70B) + Streamlit
```

- **Framework**: LangChain for RAG orchestration
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: `all-MiniLM-L6-v2` (local, 384-dim)
- **LLM**: Llama 3.3 70B Versatile via Groq
- **Interface**: Streamlit for interactive chat

## Local Setup
```bash
# Clone repository
git clone https://github.com/yordan-gergov01/science-based-personal-trainer-assistant.git
cd fitness-coach-rag

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" > .env

# Run app
streamlit run app.py
```

## 📁 Project Structure
```
├── src/
│   ├── rag_chain.py        # RAG pipeline
│   ├── config.py           # Configuration
│   └── evaluation.py       # Metrics & testing
├── notebooks/
│   ├── data_exploration.ipynb
│   └── evaluation.ipynb    # Full evaluation results
├── app.py                  # Streamlit UI
└── requirements.txt
```

## 📚 Evaluation & Analysis

Comprehensive evaluation notebooks available:
- **Data Exploration**: Document distribution, category analysis
- **Retrieval Metrics**: Precision@K, Recall@K, MRR, NDCG
- **End-to-End Testing**: Response quality, latency benchmarks

See `notebooks/` for detailed results and methodology.
