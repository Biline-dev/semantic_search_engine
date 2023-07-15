## Semantic search engine

This project aims to create a search engine that generates a list of the top k movies from a database based on a given description using BERT transformer. Additionally, it allows the inclusion of other parameters such as genre and release year to further refine the search criteria. The output of the search engine provides the titles of the movies that match the given description and additional parameters.


The dataset that has been used is: [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) 
## Run the code

Use these commands to install the dependencies on a virtual environment.

```bash
python3 -m venv eden_env
```

On the new environment install :

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip3 install -r requirements_linux.txt
```

Run this command to launch the API :
```bash
uvicorn main:app
```

## Testing
Run the file **test.py** to try the API. You can find the requests
format in the json file.


## Example
#### Simulating the solution :
![Console](https://www.pixenli.com/image/tmknRv12)