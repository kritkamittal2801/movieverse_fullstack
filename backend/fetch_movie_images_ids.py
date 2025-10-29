import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

INPUT_FILE = "data/movies_with_embeddings.pkl" 
OUTPUT_FILE = "data/movies_full.pkl"            
DEFAULT_IMAGE = "static/img/default.jpg"

# Load movies pickle
with open(INPUT_FILE, "rb") as f:
    movies_df = pd.read_pickle(f)

images = []
tmdb_ids = []

def fetch_tmdb_data(title, retries=5, delay=1):
    """Fetch TMDb poster and ID for a movie title."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data['results']:
                first = data['results'][0]
                poster_path = first.get('poster_path', '')
                tmdb_id = first.get('id', None)
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else DEFAULT_IMAGE
                return poster_url, tmdb_id
            return DEFAULT_IMAGE, None
        except requests.exceptions.RequestException:
            print(f"Attempt {attempt + 1} failed for '{title}', retrying...")
            time.sleep(delay)
    print(f"Failed to fetch TMDb data for '{title}', using default.")
    return DEFAULT_IMAGE, None

# Fetch poster + ID for all movies
for idx, title in enumerate(movies_df['title']):
    print(f"Fetching TMDb data for {idx+1}/{len(movies_df)}: {title}")
    poster_url, tmdb_id = fetch_tmdb_data(title)
    images.append(poster_url)
    tmdb_ids.append(tmdb_id)
    time.sleep(0.25)  # avoid rate limit

# Add new columns
movies_df['image'] = images
movies_df['tmdb_id'] = tmdb_ids

# Save updated DataFrame
with open(OUTPUT_FILE, "wb") as f:
    movies_df.to_pickle(f)

print(f" Saved movies with TMDb IDs and images to {OUTPUT_FILE}")
