from flask import Flask, request, session, redirect, url_for, render_template,jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import csv
import os
from huggingface_hub import HfApi, HfFileSystem
import io

from database import init_db
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import requests
from io import BytesIO
from search_api import search_bp
from huggingface_hub import HfApi
load_dotenv()

api = HfApi()
api.whoami(token=os.getenv("HF_TOKEN"))
HF_USERDATA_REPO = os.getenv("USERDATA_REPO")

fs = HfFileSystem(token=os.getenv("HF_TOKEN"))




# --- Load dataset from Hugging Face dynamically ---
HF_URL = os.getenv("MOVIES_URL", "https://huggingface.co/datasets/kritikamittal2801/movierverse-data/resolve/main/movies_full.pkl")

try:
    print(f"Loading movies from {HF_URL} ...")

    hf_token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    response = requests.get(HF_URL, headers=headers)
    response.raise_for_status()

    movies_df = pickle.load(BytesIO(response.content))
    print(f" Loaded {len(movies_df)} movies from Hugging Face (private access OK)")

except Exception as e:
    print(" Failed to load dataset:", e)
    movies_df = pd.DataFrame(columns=['title', 'description', 'image', 'embedding', 'genre'])


# Stack embeddings for similarity calculations
embs = np.vstack(movies_df['embedding'].values)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY","fallback-secret-for-local")


app.register_blueprint(search_bp, url_prefix="/api")

# Load movie data
movies = movies_df  # movies_df is loaded from movies_with_images.pkl


# Ensure users.csv exists
if not os.path.exists('users.csv'):
    with open('users.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'email', 'password'])

import math

def get_trending_movies(top_n=7):
    """Return trending/popular movies based on frequency in user_activity.csv"""
    activity_file = "user_activity.csv"
    if not os.path.exists(activity_file):
        return movies_df.sample(top_n)[['title','description','image']].to_dict(orient='records')

    df = pd.read_csv(activity_file)
    if df.empty or 'movie' not in df.columns:
        return movies_df.sample(top_n)[['title','description','image']].to_dict(orient='records')

    counts = df['movie'].value_counts().head(top_n).index.tolist()
    trending = movies_df[movies_df['title'].isin(counts)][['title','description','image']]
    # Preserve order by counts
    trending = trending.set_index('title').loc[counts].reset_index()
    return trending.to_dict(orient='records')


def generate_personalized_recommendations(movie_titles, top_n=7, max_history=10):
    """Recency + genre-aware personalized recommendations."""
    if not movie_titles:
        return get_trending_movies(top_n)

    recent = movie_titles[-max_history:]
    emb_list = []
    valid_titles = []
    genre_counts = {}

    for t in recent:
        row = movies_df[movies_df['title'] == t]
        if not row.empty and pd.notna(row.iloc[0]['embedding']).all():
            emb_list.append(row.iloc[0]['embedding'])
            valid_titles.append(t)

            # track genres
            genres = str(row.iloc[0].get('genre', '')).lower().split(', ')
            for g in genres:
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

    if not emb_list:
        return get_trending_movies(top_n)

    emb_array = np.vstack(emb_list)
    n = len(valid_titles)
    decay_base = 0.7
    weights = np.array([decay_base ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()

    mean_emb = np.average(emb_array, axis=0, weights=weights).reshape(1, -1)
    sims = cosine_similarity(mean_emb, embs)[0]
    top_indices = sims.argsort()[::-1]

    seen = set(valid_titles)
    recs = []
    for i in top_indices:
        title = movies_df.iloc[i]['title']
        if title in seen:
            continue

        genres = str(movies_df.iloc[i].get('genre', '')).lower().split(', ')
        genre_boost = sum(genre_counts.get(g, 0) for g in genres)
        score = sims[i] + 0.2 * genre_boost  # combine similarity + genre frequency

        recs.append({
            "title": title,
            "description": movies_df.iloc[i]['description'],
            "image": movies_df.iloc[i]['image'],
            "tmdb_id": str(movies_df.iloc[i].get('tmdb_id', '')),
            "score": float(score)
        })

        if len(recs) >= top_n:
            break

    recs = sorted(recs, key=lambda x: x['score'], reverse=True)[:top_n]
    return recs


def generate_similar_movies(movie_title, top_n=7):
    """Return top_n movies similar to movie_title (using genre + embedding)"""
    idx = movies_df[movies_df['title'] == movie_title].index
    if idx.empty:
        return get_trending_movies(top_n)

    movie_row = movies_df.iloc[idx[0]]
    emb = embs[idx[0]].reshape(1, -1)
    sims = cosine_similarity(emb, embs)[0]

    # Extract genre info for boosting
    movie_genres = str(movie_row.get('genre', '')).lower().split(', ')

    results = []
    for i, score in enumerate(sims):
        if i == idx[0]:
            continue  # skip the movie itself

        other_row = movies_df.iloc[i]
        other_genres = str(other_row.get('genre', '')).lower().split(', ')
        common_genres = len(set(movie_genres) & set(other_genres))

        # Boost if they share genres
        if common_genres > 0:
            score += 0.5 * common_genres  # small boost per matching genre
        else:
            score -= 0.5  # slight penalty if completely different

        results.append({
            "title": other_row['title'],
            "description": other_row['description'],
            "image": other_row['image'],
            "tmdb_id": str(other_row.get('tmdb_id', '')),
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    return results

def get_genre_weighted_recommendations(base_movie_title, top_n=10):
    """Recommend movies mainly by genre overlap, and secondarily by embedding similarity."""
    base_row = movies_df[movies_df['title'].str.lower() == base_movie_title.lower()]
    if base_row.empty:
        return []

    base_idx = base_row.index[0]
    base_emb = embs[base_idx].reshape(1, -1)
    base_genres = str(base_row.iloc[0].get('genre', '')).split()

    # Compute embedding similarities
    sims = cosine_similarity(base_emb, embs)[0]

    genre_scores = []
    for i, row in movies_df.iterrows():
        if i == base_idx:
            continue
        other_genres = str(row.get('genre', '')).split()
        genre_overlap = len(set(base_genres) & set(other_genres))

        # Strong boost for shared genres, smaller role for embeddings
        genre_boost = 0.9 * genre_overlap      
        embedding_weight = 0.4                
        score = embedding_weight * sims[i] + genre_boost

        genre_scores.append((i, score))

    # Sort and pick top results
    top_indices = [idx for idx, _ in sorted(genre_scores, key=lambda x: x[1], reverse=True)[:top_n]]

    recs = []
    for i in top_indices:
        recs.append({
            "title": movies_df.iloc[i]['title'],
            "description": movies_df.iloc[i]['description'],
            "image": movies_df.iloc[i]['image'],
            "tmdb_id": str(movies_df.iloc[i].get('tmdb_id', ''))
        })
    return recs


# Default route
@app.route('/')
def home():
    return "App is running!"



@app.route("/moviewebsite")
def movie_website():
    name = session.get("name")
    email = session.get("email")

    last_watched = None
    try:
       
        df = pd.read_csv("user_activity.csv")
        user_rows = df[df["email"] == email]
        if not user_rows.empty:
            # Get last valid movie the user watched
            last_valid = user_rows[user_rows["movie"].notna() & (user_rows["movie"] != "undefined")]
            if not last_valid.empty:
                last_watched = last_valid.iloc[-1]["movie"]
    except Exception as e:
        print("Error reading activity:", e)

    return render_template("moviewebsite.html", last_watched="Inception")


def normalize_title(title):
    # remove all non-alphanumeric characters, lowercase
    return ''.join(c for c in title.lower() if c.isalnum())

@app.route("/recommendations")
def recommendations():

    # --- Not logged in: show random recommendations ---
    if 'email' not in session:
        recs = movies_df.sample(7)[['title', 'description', 'image']].to_dict(orient='records')
        return jsonify({
            "recommended_for_you": recs,
            "because_you_watched": []
        })

    email = session['email']

    # --- Load user activity ---
    watched_movies = []
    if os.path.exists("user_activity.csv"):
        user_activity = pd.read_csv("user_activity.csv")
        user_activity['timestamp'] = pd.to_datetime(user_activity['timestamp'])
        
        # Filter for user's clicks (watched movies)
        user_activity = user_activity[
            (user_activity['email'] == email) & (user_activity['action'] == 'click')
        ].sort_values('timestamp')

        watched_movies = user_activity['movie'].dropna().tolist()

    # --- Default random fallback ---
    recommended_for_you = movies_df.sample(7)[['title', 'description', 'image', 'tmdb_id']].to_dict(orient='records')
    because_you_watched = []

    if watched_movies:
        #  Use last movie for “Because you watched”
        last_movie = watched_movies[-1].strip().lower()
        movie_to_idx = {row['title'].strip().lower(): i for i, row in movies_df.iterrows()}

        if last_movie in movie_to_idx:
            last_idx = movie_to_idx[last_movie]
            because_you_watched = get_genre_weighted_recommendations(
                movies_df.iloc[last_idx]['title'],
                top_n=11
            )

        #  Use last few (up to 10) for “Recommended for you”
        recent_movies = watched_movies[-10:]
        recommended_for_you = generate_personalized_recommendations(recent_movies, top_n=7)

    # --- Final JSON response ---
    return jsonify({
        "recommended_for_you": recommended_for_you,
        "because_you_watched": {
            "last_watched": movies_df.iloc[last_idx]['title'] if watched_movies else None,
            "recommendations": because_you_watched
        }
    })




@app.route('/payment')
def payment_page():
    return render_template('payment_page.html')

@app.route('/video')
def video():
    return render_template('video.html')

# Signup route
@app.route('/signup', methods=['GET' , 'POST'])
def signup():
    data = request.form
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    hashed_password = generate_password_hash(password)
    users_df = pd.read_csv('users.csv')

    if email in users_df['email'].values:
        return render_template('index.html', error="Email already registered")

    # Append new user to CSV
    with open('users.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, email, hashed_password])

    return render_template('index.html', success="Signup successful. Please login.")

# Login route
@app.route('/login', methods=['POST'])
def login():
    data = request.form
    email = data.get('email')
    password = data.get('password')

    users_df = pd.read_csv('users.csv')
    user_row = users_df[users_df['email'] == email]

    if user_row.empty:
        return render_template('index.html', error="User not found")

    hashed_password = user_row.iloc[0]['password']
    if check_password_hash(hashed_password, password):
        session['email'] = email
        session['name'] = user_row.iloc[0]['name']
        
            # Get the user's last watched/searched movie from user_activity.csv
        last_watched = None
        activity_file = "user_activity.csv"
        if os.path.exists(activity_file):
        
            user_activity = pd.read_csv(activity_file)
            user_logs = user_activity[user_activity['email'] == email]
            if not user_logs.empty:
                valid_logs = user_logs[user_logs['movie'].notna() & (user_logs['movie'] != 'undefined')]
                if not valid_logs.empty:
                    last_watched = valid_logs.iloc[-1]['movie']


        return render_template(
            'moviewebsite.html',
            name=user_row.iloc[0]['name'],
            last_watched=last_watched
        )


    else:
        return render_template('index.html', error="Incorrect password")

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return render_template('index.html')


# Movies route
@app.route('/movies', methods=['GET'])
def get_movies():
    return movies.to_json(orient='records')

from datetime import datetime
from flask import request, jsonify
import csv

@app.route('/activity', methods=['POST'])
def activity():
    if 'email' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    data = request.get_json()
    action = data.get('action')
    movie = data.get('movie')

    if not action or not movie:
        return jsonify({'status': 'error', 'message': 'Missing action or movie'}), 400

    file_path = 'user_activity.csv'
    file_exists = os.path.exists(file_path)

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'email', 'name', 'action', 'movie'])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session.get('email'),
            session.get('name'),
            action,
            movie
        ])

    return jsonify({'status': 'success'})


@app.route('/health')
def health():
    return "OK", 200

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

