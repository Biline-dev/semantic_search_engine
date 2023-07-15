import random
import json
import requests
import pprint

# Endpoint of the API
url="http://127.0.0.1:8000/search_movie/"



# Function to get a random request from a JSON file
def get_random_payload(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    # Convert the dictionary keys to a list
    payload_keys = list(data.keys())
    # Randomly select a payload key
    selected_payload_key = random.choice(payload_keys)
    # Get the selected payload using the key
    selected_payload = data[selected_payload_key]

    return selected_payload


# Test the API
request = get_random_payload("test.json")

response = requests.post(url, params=request)
# Handle the response
if response.status_code == 200:
    # Request successful
    list_movies = response.json()
    print('\n')
    print("******** Given request: ********")
    pprint.pprint(request)
    print('\n')
    print("******** Recommended movies: ********")
    pprint.pprint(list_movies)
    print('\n')
else:
    # Request failed
    print("Error:", response.status_code)
