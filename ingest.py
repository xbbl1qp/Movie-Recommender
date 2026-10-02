from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

###-----------Configuration----------------###
DATA_DIR = Path("data/ml-latest-small")
CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "movies"

EMBEDDING_MODEL_NAME = ("sentence-transformers/all-MiniLM-L6-v2")

BATCH_SIZE = 256

#----------------Load & Prepare Data---------#

def load_movies():
    movies_path = DATA_DIR/ "movies.csv"
    tags_path = DATA_DIR/ "tags.csv"

    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    # Combine all tags belong to the same movie
    tags_by_movie = (
        tags.groupby("movieId")["tag"]
        .apply(
            lambda values: ", ".join(
                sorted(
                    set(
                        str(value)
                        for value in values.dropna()
                    )
                )
            )
        ).reset_index()
    )

    tags_by_movie = tags_by_movie.rename(
        columns = {"tag": "tags"}
    )

    # keep every movie, even if no tags
    movies = movies.merge(
       tags_by_movie,
       on="movieId",
       how="left"
    )

    movies["tags"] = movies["tags"].fillna("")

    #Contruct the text miniLM will understand
    movies["document"] = (
        "Title: " + movies["title"].fillna("")
        +", Genres: "
        + movies["genres"]
            .fillna("")
            .str.replace("|",", ", regex =False)
        + ". Tags: "
        + movies["tags"]
    )

    return movies
   

    #--------------Create Vector Database--------#

def create_collection(client):
    try:
            client.delete_collection(COLLECTION_NAME)
    except Exception as error:
            if "does not exist" not in str(error).lower():
                raise
    return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space":"cosine"}
        )

#---------------Main Ingestion JOb-----------#
def main():
    print("Loading Movie Data.....")

    movies = load_movies()

    print(f"MOvies Loaded : {len(movies)}")

    print("Loading Embedding Model...")

    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(
            path= str(CHROMA_DIR)
    )    

    collection = create_collection(client)

    print("Creating and Storing embeddings")

    for start in range(0, len(movies), BATCH_SIZE):
        batch = movies.iloc[
            start:start + BATCH_SIZE
        ]

        documents = batch["document"].tolist()

         # Generate Vector for this batch

        embeddings = encoder.encode(
            documents,
            show_progress_bar = False
        )

        # Chrome Ids must be String

        ids= [
            str(movie_id)
            for movie_id in batch["movieId"]
        ]

        #Metadata Values must be supported Scalar Type

        metadatas = [
            {
                "movie_id": int(row["movieId"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "tags": str(row["tags"])
            }
                for _, row in batch.iterrows()
        ]

        collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
        )

        print(
            f"Indexed {min(start + BATCH_SIZE, len(movies))}"
            f"/ {len(movies)} movies"
        )

    print("Ingestion Complete.. ")
    print("Stored Records:", collection.count())

if __name__ == "__main__":
    main()