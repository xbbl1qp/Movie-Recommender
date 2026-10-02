from sentence_transformers import SentenceTransformer

#Load Model which you need for embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [ "A science-fiction film about space and time.",
    "A romantic comedy about two people falling in love.",
    "A mysterious science-fiction story about aliens."
]

#Embeding for the sentences
embeddings = model.encode(sentences)

print("Embedding :", embeddings.shape)

similarity = model.similarity(
    embeddings[0:1],
    embeddings[2:3]
)

print("Similarity:", similarity.item())