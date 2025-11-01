# fetch_tmdb.py
import requests
import csv
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"
OUT_CSV = "./data/movies_raw.csv"

def fetch_popular_movies(pages=50):
    movies = []
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        success = False
        for attempt in range(3):  # retry up to 3 times
            try:
                print(f"Fetching page {page} (attempt {attempt+1})...")
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()
                for m in data.get("results", []):
                     # Extract genre names from TMDb genre IDs
                     genre_names = []
                     if "genre_ids" in m and m["genre_ids"]:
                            genre_names = [str(gid) for gid in m["genre_ids"]]  # fallback, IDs for now

                     movies.append({
                            "title": m.get("title"),
                            "description": m.get("overview"),
                            "release_date": m.get("release_date"),
                            "genre": ", ".join(genre_names)
                })

                print(f" Page {page} fetched successfully ({len(movies)} movies total)")
                success = True
                time.sleep(2)  # wait to avoid being blocked
                break
            except requests.exceptions.RequestException as e:
                print(f" Error fetching page {page}: {e}")
                time.sleep(5)
        if not success:
            print(f"Failed to fetch page {page} after 3 attempts, skipping...")
    return movies



if __name__ == "__main__":
    movies = fetch_popular_movies(pages=50)  
    with open(OUT_CSV, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "description", "release_date", "genre"])
        writer.writeheader()
        for m in movies:
            writer.writerow(m)
