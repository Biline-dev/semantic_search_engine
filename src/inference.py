
import numpy as np

from scipy.spatial.distance import cdist


class SearchEngine:

    """
    This class is used to perform the research given a request
    """

    def __init__(self, model, tokenizer, data, list_years, list_genre, k=5, 
                 genre = None, release_year = None, query = None):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.list_years = list_years
        self.list_genre = list_genre
        self.k = k
        self.genre = genre
        self.release_year = release_year
        self.query = query


    def calculate_similarity(self, filtered_df):

        desired_columns = ['title', 'release_year', 'genre']

        if self.query!= None:
        
            # Get query embeddings
            encoded_input = self.tokenizer.encode(self.query,  max_length=125, padding='max_length', truncation=True, return_tensors='pt')
            output = self.model(encoded_input)
            embedding_query = output.last_hidden_state.mean(dim=1).detach().numpy()
            # Convert text embeddings to a numpy array
            texts_encoded = list(filtered_df['encoded_text'])
            text_embeddings = np.array(texts_encoded)

            # calculate similarity between the query and all rows of filtered_df
            similarities = 1 - cdist(text_embeddings, embedding_query, metric='cosine').flatten()
  
            # the list is after added to the dataframe
            filtered_df['similarities'] = similarities

            # This method allow dataframe to take k best movies and return a sorted dataframe
            filtered_df = filtered_df.nlargest(self.k, 'similarities')
            
            return filtered_df[desired_columns]
        
        else: 
            return filtered_df[desired_columns]
        


    def search(self):

        if self.release_year != None:  
            if self.release_year in self.list_years:
                if self.genre != None:
                    if self.genre in self.list_genre:
                        ### case one 1 :  we search (release_year, genre)
                        filtered_df = self.data[self.data['year_genre'] == (self.release_year, self.genre)]
                        return self.calculate_similarity(filtered_df)
                    else:
                        raise Exception("Genre not found")
                else:
                    ### case one 2 :  we search (release_year, all)
                    filtered_df = self.data[self.data['release_year'] == self.release_year]
                    return self.calculate_similarity(filtered_df)   
            else:
                raise Exception("Year not found")
        else:
            if self.genre != None:
                if self.genre in self.list_genre:
                    ### case one 3 : we search (all, genre)
                    filtered_df = self.data[self.data['genre'] == self.genre]
                    return self.calculate_similarity(filtered_df)
                else:
                    raise Exception("Genre not found")
            else:
                ### case one 4 :  we search (all, all)
                filtered_df = self.data
                return self.calculate_similarity(filtered_df) 

    def return_dict(self, new_data):

        suggestions_dict = {}
        for index, row in new_data.iterrows():
            suggestion_key = f"movie {index + 1}"
            suggestion_value = {
                "Title": row['title'],
                "year": row['release_year'],
                "genre": row['genre']
            }
            suggestions_dict[suggestion_key] = suggestion_value

        return suggestions_dict
        