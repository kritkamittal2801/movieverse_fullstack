from flask import Flask, request, session, redirect, url_for, render_template,jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import os
from huggingface_hub import HfApi, HfFileSystem
import io
from database import upload_db_to_hf
from flask import redirect, url_for
from database import download_db_from_hf, init_db, SessionLocal, User, UserActivity
from sqlalchemy.orm import Session
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
import math
from io import BytesIO
from database import SessionLocal, UserActivity
from search_api import search_bp
from huggingface_hub import HfApi
load_dotenv()
import sys
print(">>> Running app.py from:", os.path.abspath(__file__))
print(">>> Current working directory:", os.getcwd())

print(" Starting Flask app...", file=sys.stderr)

api = HfApi()
api.whoami(token=os.getenv("HF_TOKEN"))
HF_USERDATA_REPO = os.getenv("USERDATA_REPO")

fs = HfFileSystem(token=os.getenv("HF_TOKEN"))

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
app.secret_key = os.getenv("SECRET_KEY","fallback-secret-for-local")
app.register_blueprint(search_bp, url_prefix="/api")
@app.route('/test')
def test_page():
    return render_template('index.html')

download_db_from_hf()
init_db()
#  --- Load dataset from Hugging Face dynamically ---
HF_URL = os.getenv("MOVIES_URL", "https://huggingface.co/kritikamittal2801/movierverse-data/resolve/main/movies_full.pkl")

try:
    print(" Step 1: Starting to load movies from HF...")
    hf_token = os.getenv("HF_TOKEN")

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    print(" Step 2: Sending request to Hugging Face...")
    response = requests.get(HF_URL, headers=headers)
    print(f" Step 3: Got response ({len(response.content)} bytes). Now unpickling...")

    response.raise_for_status()
    movies_df = pickle.load(BytesIO(response.content))
    print(f"Step 4: Loaded {len(movies_df)} movies from Hugging Face successfully!")

except Exception as e:
    print(" Failed to load dataset:", e)
    movies_df = pd.DataFrame(columns=['title', 'description', 'image', 'embedding', 'genre'])

# Stack embeddings for similarity calculations
embs = np.vstack(movies_df['embedding'].values)

# Load movie data
movies = movies_df  # movies_df is loaded from movies_with_images.pkl


def get_trending_movies(top_n=7):
    """Return trending/popular movies based on frequency in UserActivity table."""
    db = SessionLocal()
    
    try:
        # Get most watched movies (action = 'click')
        activities = (
            db.query(UserActivity.movie)
            .filter(UserActivity.action == "click")
            .all()
        )
        
        # Extract movie titles from query results
        movie_list = [a.movie for a in activities if a.movie]

        if not movie_list:
            return movies_df.sample(top_n)[['title','description','image']].to_dict(orient='records')

        # Count frequencies
        movie_counts = pd.Series(movie_list).value_counts().head(top_n)
        top_titles = movie_counts.index.tolist()

        # Filter and preserve order
        trending = movies_df[movies_df['title'].isin(top_titles)][['title','description','image']]
        trending = trending.set_index('title').loc[top_titles].reset_index()
        
        return trending.to_dict(orient='records')

    except Exception as e:
        print("Error fetching trending movies:", e)
        return movies_df.sample(top_n)[['title','description','image']].to_dict(orient='records')

    finally:
        db.close()



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

        unique_titles = set()
        unique_recs = []
        for r in recs:
            t = r['title'].strip().lower()
            if t not in unique_titles:
                unique_titles.add(t)
                unique_recs.append(r)

        recs = unique_recs
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
    
    # remove duplicate titles 
    unique_titles = set()
    unique_recs = []
    for r in recs:
        t = r['title'].strip().lower()
        if t not in unique_titles:
            unique_titles.add(t)
            unique_recs.append(r)

    return unique_recs


# Default route
@app.route("/")
def home():
    if 'email' in session:
        return redirect(url_for('movie_website1'))
    print("Rendering index.html...", flush=True)
    return render_template('index.html')

@app.route("/moviewebsite")
def movie_website():
    if 'email' not in session:
        return redirect(url_for('home'))  # redirect to login if not logged in
    name = session.get("name")
    email = session.get("email")

    db = SessionLocal()
    last_watched = None

    try:
        # Find the logged-in user
        user = db.query(User).filter(User.email == email).first()

        if user:
            # Get latest watched movie for this user
            activity = (
                db.query(UserActivity)
                .filter(UserActivity.user_id == user.id, UserActivity.action == "click")
                .order_by(UserActivity.timestamp.desc())
                .first()
            )

            if activity:
                last_watched = activity.movie

    except Exception as e:
        print("Error reading activity from DB:", e)

    finally:
        db.close()

    # If no movie found, fallback to a default
    return render_template("moviewebsite.html", last_watched=last_watched or "Inception")


def normalize_title(title):
    # remove all non-alphanumeric characters, lowercase
    return ''.join(c for c in title.lower() if c.isalnum())

@app.route('/create_new_profile')
def create_new_profile():
    session.clear()  # same as logout
    return redirect(url_for('home'))  # goes to index.html (login/signup)

@app.route("/recommendations")
def recommendations():
    db = SessionLocal()
    # --- Not logged in: show random recommendations ---
    if 'email' not in session:
        recs = movies_df.sample(7)[['title', 'description', 'image']].to_dict(orient='records')
        return jsonify({
            "recommended_for_you": recs,
            "because_you_watched": []
        })

    email = session['email']

    # --- Load user activity ---
    user = db.query(User).filter(User.email == session['email']).first()
    user_activity = (
    db.query(UserActivity)
    .filter(UserActivity.user_id == user.id, UserActivity.action == 'click')
    .order_by(UserActivity.timestamp)
    .all()
)
    watched_movies = [a.movie for a in user_activity]


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
    db.close()
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
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    db = SessionLocal()
    data = request.form
    username = data.get('name')
    email = data.get('email')
    password = data.get('password')

    # Check if user exists
    existing_user = db.query(User).filter((User.email == email) | (User.username == username)).first()
    if existing_user:
        return render_template('index.html', error="Email or username already registered")

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    db.close()
    try:
        upload_db_to_hf()
    except Exception as e:
        print("Warning: failed to upload DB after signup:", e)

    return render_template('index.html', success="Signup successful. Please login.")


# Login route
@app.route('/login', methods=['POST'])
def login():
    db = SessionLocal()
    data = request.form
    email = data.get('email')
    password = data.get('password')

    user = db.query(User).filter(User.email == email).first()
    if not user:
        db.close()
        return render_template('index.html', error="User not found")

    if check_password_hash(user.password, password):
        session['email'] = user.email
        session['name'] = user.username

        db.close()
        return redirect(url_for('movie_website'))

    db.close()
    return render_template('index.html', error="Incorrect password")

@app.route('/movie')
def movie_website1():
    if 'email' not in session:
        return redirect(url_for('home'))

    db = SessionLocal()
    user = db.query(User).filter(User.email == session['email']).first()

    last_activity = (
        db.query(UserActivity)
        .filter(UserActivity.user_id == user.id, UserActivity.action == "click")
        .order_by(UserActivity.timestamp.desc())
        .first()
    )
    last_watched = last_activity.movie if last_activity else None

    db.close()
    return render_template('moviewebsite.html', name=user.username, last_watched=last_watched)


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('home'))

@app.route('/movies', methods=['GET'])
def get_movies():
    return movies.to_json(orient='records')

from datetime import datetime
from flask import request, jsonify

@app.route('/activity', methods=['POST'])
def activity():
    db = SessionLocal()
    if 'email' not in session:
        return jsonify({'status': 'error', 'message': 'User not logged in'}), 401

    data = request.get_json()
    action = data.get('action')
    movie = data.get('movie')

    if not action or not movie:
        return jsonify({'status': 'error', 'message': 'Missing action or movie'}), 400

    user = db.query(User).filter(User.email == session['email']).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'}), 404

    activity = UserActivity(user_id=user.id, action=action)
    setattr(activity, "movie", movie)
    db.add(activity)
    db.commit()
    db.refresh(activity)
    db.close()
    try:
        upload_db_to_hf()
    except Exception as e:
        print("Warning: failed to upload DB after activity:", e)

    return jsonify({'status': 'success'})


@app.route('/health')
def health():
    return "OK", 200

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)

