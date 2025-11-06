from flask import Blueprint, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import re
import requests
from io import BytesIO
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- TMDB Genre ID → Name mapping ---
GENRE_MAP = {
    28: "action", 12: "adventure", 16: "animation", 35: "comedy",
    80: "crime", 99: "documentary", 18: "drama", 10751: "family",
    14: "fantasy", 36: "history", 27: "horror", 10402: "music",
    9648: "mystery", 10749: "romance", 878: "sci-fi",
    10770: "tv movie", 53: "thriller", 10752: "war", 37: "western"
}

def map_genres(genre_str):
    """Convert TMDB genre IDs (like '10749 18') to names ('romance, drama')."""
    if pd.isna(genre_str): 
        return ""
    ids = []
    for part in str(genre_str).split():
        try:
            ids.append(int(part))
        except:
            continue
    names = [GENRE_MAP.get(i, str(i)) for i in ids]
    return ", ".join(names)


search_bp = Blueprint('search', __name__)
CORS(search_bp)

HF_URL = os.getenv("MOVIES_URL", "https://huggingface.co/kritikamittal2801/movierverse-data/resolve/main/movies_full.pkl")

try:
    print(f"Loading search data from {HF_URL} ...")

    hf_token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    response = requests.get(HF_URL, headers=headers)
    response.raise_for_status()

    movies_df = pickle.load(BytesIO(response.content))
    print(f" Search dataset loaded ({len(movies_df)} movies)")

except Exception as e:
    print(" Failed to load search dataset:", e)
    movies_df = pd.DataFrame(columns=['title', 'description', 'genre', 'image', 'embedding', 'release_year'])



# Stack embeddings into a NumPy array
embs = np.vstack(movies_df["embedding"].values)

# Convert numeric genre IDs to readable genre names
movies_df["genre"] = movies_df["genre"].apply(map_genres)


# Ensure release_year exists as integer
if 'release_year' not in movies_df.columns and 'release_date' in movies_df.columns:
    movies_df['release_year'] = pd.to_numeric(
        movies_df['release_date'].astype(str).str[:4], errors='coerce'
    )
movies_df['release_year'] = pd.to_numeric(movies_df.get('release_year', pd.Series(dtype='float')), errors='coerce').astype('Int64')

# Ensure embeddings are numpy arrays
try:
    embs = np.vstack(movies_df['embedding'].values)
except Exception as e:
    raise RuntimeError("Failed to stack embeddings from movies_full.pkl") from e

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Search model loaded successfully! Embedding dim:", embs.shape[1])

@search_bp.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])  # frontend expects array

        # Detect if query refers to a specific genre
        # --- Detect if query refers to one or more specific genres ---
    GENRE_KEYWORDS = {
        "romance": ["romance", "love", "love story", "romantic", "affair"],
        "comedy": ["comedy", "funny", "humor", "hilarious"],
        "action": ["action", "fight", "battle", "hero"],
        "thriller": ["thriller", "suspense", "mystery"],
        "horror": ["horror", "scary", "fear", "terror"],
        "drama": ["drama", "emotional", "tearjerker"],
        "sci-fi": ["sci-fi", "science fiction", "space"],
        "fantasy": ["fantasy", "magic", "myth"],
        "crime": ["crime", "gangster", "heist"],
        "adventure": ["adventure", "journey", "exploration"]
    }

    detected_genres = []
    for g, synonyms in GENRE_KEYWORDS.items():
        if any(word in q.lower() for word in synonyms):
            detected_genres.append(g)



    # Extract year from query if present
    query_years = re.findall(r'\b(?:19|20)\d{2}\b', q)
    exact_year_df = pd.DataFrame()
    plus_minus_year_df = pd.DataFrame()

    if query_years:
        query_year = int(query_years[0])

        # Exact year first
        exact_year_df = movies_df[movies_df['release_year'] == query_year]

        # Year ±1
        plus_minus_year_df = movies_df[
            (movies_df['release_year'] >= query_year - 1) &
            (movies_df['release_year'] <= query_year + 1) &
            (movies_df['release_year'] != query_year)
        ]

        # Combine exact + ±1 (exact first)
        filtered_df = pd.concat([exact_year_df, plus_minus_year_df])
    else:
        # No year in query → search all movies
        filtered_df = movies_df

    # If no movies match year criteria, return empty
    if filtered_df.empty:
        return jsonify([])

    # Convert embeddings safely
    try:
        filtered_embs = np.vstack(filtered_df['embedding'].values)
    except:
        # fallback to global embeddings if something wrong
        filtered_embs = embs
        filtered_df = movies_df

    # Compute query embedding
    q_emb = np.asarray(model.encode(q.lower())).reshape(1, -1)
    if q_emb.shape[1] != filtered_embs.shape[1]:
        return jsonify({"error": "Embedding dimension mismatch"}), 500

    sims = cosine_similarity(q_emb, filtered_embs)[0]

    # Build results with boosts
    results = []
    for idx, score in enumerate(sims):
        row = filtered_df.iloc[idx]
        title = str(row.get('title', ''))
        genre = str(row.get('genre', '')).lower()
        description = str(row.get('description', ''))
        year = str(int(row['release_year'])) if pd.notna(row.get('release_year')) else ""

               # --- Improved SMART BOOST LOGIC ---
        q_lower = q.lower()
        title_lower = title.lower()
        desc_lower = description.lower()
        genre_lower = str(genre).lower()

        title_match = q_lower in title_lower
        desc_match = q_lower in desc_lower

        #  Multi-genre logic
        if detected_genres:
            matched_any = any(g in genre_lower for g in detected_genres)
            matched_all = all(g in genre_lower for g in detected_genres)

            if matched_all:
                score += 6.0  # very strong boost if all detected genres match
            elif matched_any:
                score += 3.0  # moderate boost if some genres match
            else:
                score -= 2.0  # penalty if wrong genre
        elif title_match:
            score += 2.0
        elif desc_match:
            score += 0.5


        results.append({
    "title": title,
    "year": year,
    "description": description,
    "genre": row.get('genre', ''),
    "image": row.get('image', ''),
    "tmdb_id": str(row.get('tmdb_id', '')),
    "score": float(score)
})


    # Sort by score and limit to 5
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]

    return jsonify(results)
