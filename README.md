# Movie-Recommender
Movie recommender will recommend movie based on the user input it is using RAG, Vector Database internally to do so.

Steps to run :

Install Python 3.11.x
 
py -3.11 -m venv .venv

source .venv/Scripts/activate

pip install -r requirements.txt

### Ingest Movie Data ###
python ingest.py

### Movie Recommender ###
python recommender.py
