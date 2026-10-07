# Movie-Recommender
 # Movie recommender will recommend movie based on the user input it is using RAG Architecure.
 # Movies are recommend from dataset which i took from movie lens.
 # I have used Chroma Vector Database internally to store emnbeddings of movie data points.


 Dataset Used : https://grouplens.org/datasets/movielens/ 
 Embedding Model : sentence-transformers/all-MiniLM-L6-v2
 LLM Used : Qwen/Qwen2.5-0.5B-Instruct

Steps to run :

Install Python 3.11.x
 
py -3.11 -m venv .venv

source .venv/Scripts/activate

pip install -r requirements.txt

### Ingest Movie Data ###
python ingest.py
