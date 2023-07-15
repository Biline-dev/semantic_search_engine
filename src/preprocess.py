import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from transformers import BertTokenizer, BertModel


"""

This code has been used to encode the text of the plots of the movies using the BERT model.
The created embeddings are added to the dataframe and saved as a numpy array.
The aim of this approach is to speed up the search of similar movies by preparing the data beforehand.

This code is not used in the final version of the project and was mostly run on GoogleColab.


"""

# Load the data
DF = pd.read_csv('../data/wiki_movie_plots_deduped.csv')


# Drop columns that are not needed
columns_to_drop = ['Director', 'Cast', 'Wiki Page', 'Origin/Ethnicity']
DF = DF.drop(columns=columns_to_drop, axis=1)

# Rename columns
DF = DF.rename(columns={"Release Year": "release_year", "Title": "title", "Genre": "genre", "Plot": "plot"})

# Add a column with the year and genre together that will be used to speed up the search
DF['year_genre'] = list(zip(DF['release_year'], DF['genre']))

# Load the tokenizer and model
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertModel.from_pretrained("bert-base-uncased")



def encode_text(df = DF, tokenizer = TOKENIZER, model = MODEL):
    

    plots = list(df['plot'])  # A plot is a description of the movie
    batch_size = 100 #  batch size = 100 to prevent RAM insuffisance
    num_texts = len(plots)
    encoded_texts = []
    embeddings = []
    for i in tqdm(range(0, num_texts, batch_size)):
        batch_texts = plots[i:i+batch_size]
        encoded_batch = [tokenizer.encode(text, max_length=125, padding='max_length', truncation=True, add_special_tokens=True) for text in batch_texts]
        encoded_texts.extend(encoded_batch)


    batch_size = 10 #  batch size = 10 to prevent GPU insuffisance
    # Convert the encoded texts to tensors
    input_ids = torch.tensor(encoded_texts)

    embeddings = []
    # Generate embeddings for the query and move it to the device
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_texts, batch_size)):
            input_batch = input_ids[i:i+batch_size]
            outputs = model(input_batch)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.extend(batch_embeddings)


    list_embeddings = []

    for output in embeddings:
        list_embeddings.append(output.detach().numpy())

    df['encoded_text'] = list_embeddings

    np.save('../data/df_encoded.npy', df)

encode_text(df = DF, tokenizer = TOKENIZER, model = MODEL)