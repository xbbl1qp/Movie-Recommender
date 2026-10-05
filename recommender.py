from pathlib import Path
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
import torch 
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM)

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "movies"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"


def load_llm():
    print("Loading Mistral 7B, fist laod take time...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map = "auto",
        torch_dtype = "auto"
    )

    model.eval()

    return tokenizer, model

def generate_recommendations(query, movies, tokenizer, model):

    context = "\n".join(
        (
            f"Title : {movie['metadata']['title']}\n"
            f"Genres : {movie['metadata']['genres']}\n"
            f"Tags : {movie['metadata']['tags']}\n"
            f"Movie information: {movie['document']}"
        )
        for movie in movies
    )

    prompt = f"""[INST]
    You are a helpful movie recommendation assistant.

    The user's preference:
    {query}

    Here are the candidate movies retrieved from database:
    {context}

    Recommend up to five relevant movies from the list only.
    
    For each recmmendation:
    1. State the exact movie title.
    2. Explain why it matches the user's preference.
    3. Use only the supplied movie information.
    4. Do not invent plot details, actors, or facts
    5. If the available information is not sufficient say so.

    Do not recommend any title  that is absent from the candidate list.
    [/INST]"""

    inputs = tokenizer(
        prompt,
        return_tensors = "pt"
    ).to(model.device)

    # Generate text without calculating  training gradients
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens = 300,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id
        )

    # Remove the prompt tokens from the generated output.
    new_tokens = output[
        0, inputs["inputs_id"].shape[1]:
    ]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens = True
    )

    return answer.strip()
    


def load_retrieval_components():
    # Load the same emmbedding model used during ingestion
    encoder = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    client = chromadb.PersistentClient(
        path= str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return encoder, collection

def retrieve_movies(query, encoder, collection, limit=10):
    #Convert the user request into vector 
    query_vector = encoder.encode(
        [query]
    ).tolist()

    #Search for semantically similar movie documents.

    results = collection.query(
        query_embeddings = query_vector,
        n_results = limit,
        include = [
            "documents",
            "metadatas",
            "distances"
        ]
    )

    movies = []

    for index in range(len(results["ids"][0])):
        movies.append({
            "id": results["ids"][0][index],
            "document": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index]
        })

    return movies   

def main(): 
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Test Vector without loading Mistral"
    )

    args = parser.parse_args()

    query = input(
        "Describe the moveie you want:"
    ).strip()

    if not query:
        print("Please enter a movie preference.")
        return

    encoder, collection = load_retrieval_components()

    movies = retrieve_movies(
        query,
        encoder,
        collection,
        limit=10
    )

    print("\nRetrieved movie collection:\n")

    for rank, movie in enumerate(movies, start=1):
        metadata = movie["metadata"]

        print(
            f"{rank}. {metadata['title']}\n"
            f"  Genres: {metadata['genres']}\n"
            f"  Distance: {movie['distance']:.4f}\n"     
        )

        if args.retrieve_only:
            return
        
        tokenizer, model = load_llm()

        answer = generate_recommendations(
            query,
            movies,
            tokenizer,
            model
        )

        print("\n ===== RECOMMENDATIONS =====\n")
        print(answer)


if __name__ == '__main__':
    main()
