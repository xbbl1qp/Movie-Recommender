from pathlib import Path
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "movies"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# Load Qwen LLM
# ============================================================

def load_llm():

    print("Loading Qwen 0.5B... First load may take some time.")

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ============================================================
# Generate Recommendations
# ============================================================

def generate_recommendations(
    query,
    movies,
    tokenizer,
    model
):

    # --------------------------------------------------------
    # Build context from retrieved movies
    # --------------------------------------------------------

    context = "\n\n".join(
        (
            f"Title: {movie['metadata']['title']}\n"
            f"Genres: {movie['metadata']['genres']}\n"
            f"Tags: {movie['metadata']['tags']}\n"
            f"Movie information: {movie['document']}"
        )
        for movie in movies
    )

    # --------------------------------------------------------
    # Qwen Chat Messages
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful movie recommendation assistant.

Your job is to recommend movies ONLY from the candidate movies
provided by the user.

Rules:
1. Recommend up to five movies.
2. Use the exact movie title from the candidate list.
3. Explain why each movie matches the user's preference.
4. Use only the supplied movie information.
5. Do not invent plot details, actors, ratings, or other facts.
6. Do not recommend movies that are not in the candidate list.
7. If the information is insufficient, clearly say so.
"""
        },
        {
            "role": "user",
            "content": f"""
User preference:

{query}

Candidate movies retrieved from the vector database:

{context}

Recommend the best matching movies from the candidate list.
"""
        }
    ]

    # --------------------------------------------------------
    # Convert messages to Qwen's expected chat format
    # --------------------------------------------------------

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # --------------------------------------------------------
    # Remove the input prompt from the output
    # --------------------------------------------------------

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================
# Load Vector Database + Embedding Model
# ============================================================

def load_retrieval_components():

    print("Loading embedding model...")

    # Same embedding model used during ingestion
    encoder = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return encoder, collection


# ============================================================
# Retrieve Movies
# ============================================================

def retrieve_movies(
    query,
    encoder,
    collection,
    limit=10
):

    # --------------------------------------------------------
    # Convert user query into embedding
    # --------------------------------------------------------

    query_vector = encoder.encode(
        [query]
    ).tolist()

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=query_vector,
        n_results=limit,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # --------------------------------------------------------
    # Convert ChromaDB result into easier Python structure
    # --------------------------------------------------------

    movies = []

    for index in range(len(results["ids"][0])):

        movies.append(
            {
                "id": results["ids"][0][index],
                "document": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": results["distances"][0][index]
            }
        )

    return movies


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Test vector search without loading Qwen"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Get user query
    # --------------------------------------------------------

    query = input(
        "Describe the movie you want: "
    ).strip()

    if not query:

        print("Please enter a movie preference.")

        return

    # --------------------------------------------------------
    # Load RAG retrieval components
    # --------------------------------------------------------

    encoder, collection = load_retrieval_components()

    # --------------------------------------------------------
    # Retrieve relevant movies
    # --------------------------------------------------------

    movies = retrieve_movies(
        query,
        encoder,
        collection,
        limit=10
    )

    # --------------------------------------------------------
    # Display retrieved movies
    # --------------------------------------------------------

    print("\n===== RETRIEVED MOVIES =====\n")

    for rank, movie in enumerate(
        movies,
        start=1
    ):

        metadata = movie["metadata"]

        print(
            f"{rank}. {metadata['title']}\n"
            f"   Genres: {metadata['genres']}\n"
            f"   Tags: {metadata['tags']}\n"
            f"   Distance: {movie['distance']:.4f}\n"
        )

    # --------------------------------------------------------
    # Stop here if retrieve-only mode
    # --------------------------------------------------------

    if args.retrieve_only:

        return

    # --------------------------------------------------------
    # Load LLM
    # IMPORTANT: Load only ONCE, outside the loop
    # --------------------------------------------------------

    tokenizer, model = load_llm()

    # --------------------------------------------------------
    # Generate recommendation
    # --------------------------------------------------------

    answer = generate_recommendations(
        query,
        movies,
        tokenizer,
        model
    )

    # --------------------------------------------------------
    # Display final answer
    # --------------------------------------------------------

    print("\n===== RECOMMENDATIONS =====\n")

    print(answer)


# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":
    main()
