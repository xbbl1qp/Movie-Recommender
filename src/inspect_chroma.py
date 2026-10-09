import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection("movies")

print("Collection:", collection.name)
print("Number of movies:", collection.count())

data = collection.get(
    limit=5,
    include=[
        "documents",
        "metadatas",
        "embeddings"
    ]
)

for i in range(len(data["ids"])):
    print("\nID:", data["ids"][i])
    print("Document:", data["documents"][i])
    print("Metadata:", data["metadatas"][i])
