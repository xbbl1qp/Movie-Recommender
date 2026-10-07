# 🎬 Movie Recommender

A simple **AI-powered movie recommendation system** built using **Retrieval-Augmented Generation (RAG)**.

The application recommends movies based on the user's natural-language input. Movie information is retrieved from a **ChromaDB vector database** using semantic similarity, and the results are used by an LLM to generate recommendations.

## 🛠️ Tech Stack

* **Python 3.11**
* **RAG (Retrieval-Augmented Generation)**
* **ChromaDB** – Vector Database
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Dataset:** [MovieLens](https://grouplens.org/datasets/movielens/)

## 📊 Dataset

This project uses the **MovieLens Latest Small Dataset**.

Dataset: https://grouplens.org/datasets/movielens/

Movie information such as **title, genres, and tags** is converted into embeddings and stored in ChromaDB for semantic search.

## 🔄 RAG Flow

```text
User Query
    ↓
Generate Query Embedding
    ↓
Search ChromaDB
    ↓
Retrieve Relevant Movies
    ↓
Send Retrieved Context to LLM
    ↓
Generate Movie Recommendations
```

## 🚀 Setup & Run

### 1. Install Python 3.11.x

Check your Python version:

```bash
python --version
```

### 2. Create Virtual Environment

```bash
py -3.11 -m venv .venv
```

### 3. Activate Virtual Environment

**Git Bash:**

```bash
source .venv/Scripts/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download MovieLens Dataset

Download the dataset from:

https://grouplens.org/datasets/movielens/

Extract it under:

```text
data/ml-latest-small/
```

### 6. Ingest Movie Data

Run the ingestion process to generate embeddings and store them in ChromaDB:

```bash
python ingest.py
```

## 🔮 Future Improvements

* Query summarization
* Query routing
* Hybrid search
* Reranking
* Context compression
* RAG evaluation
* Azure deployment
* GitLab CI/CD
