import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertModel

from src.inference import SearchEngine

from fastapi import FastAPI, HTTPException

# Load the the database
data = np.load('data/wiki_movie_plots_deduped_encoded.npy', allow_pickle=True)

# Convert the data back to a dataframe
DF = pd.DataFrame(data)
DF.columns = ["release_year", "title", "genre", "plots", "year_genre", "encoded_text"]

# Load the model
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained("bert-base-uncased")

# Load the list of years and genres
LIST_YEARS = list(DF['release_year'].unique())
LIST_GENRE = list(DF['genre'].unique())



app = FastAPI()

@app.post("/search_movie/")
async def create_upload_file(query: str = None, k: int = 5, 
                             genre: str = None, release_year: int = None):
    
    
    search = SearchEngine(MODEL, TOKENIZER, DF, LIST_YEARS, LIST_GENRE, k=k, 
                            genre=genre, release_year=release_year, query=query)
    results = search.search()
    if results is None:
        raise HTTPException(status_code=404, detail="No results found")
    else:
        return search.return_dict(results)
