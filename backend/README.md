MovieVerse Backend

Flask + SentenceTransformer API for semantic movie search.

---

 Features
- Flask backend serving movie recommendation and search APIs  
- Uses SentenceTransformers for semantic similarity  
- Preprocessing and embeddings built from CSV movie data  
- Includes user activity tracking and authentication  
- Ready for deployment on Render

---

 Project Structure
```
movieverse_fullstack/
 └── backend/
     ├── app.py
     ├── search_api.py
     ├── fetch_movie_images_ids.py
     ├── data/
     ├── scripts/
     ├── static/
     ├── templates/
     ├── requirements.txt
     └── README.md
```

---

 Setup & Run Locally

 1. Clone the repository
```bash
git clone https://github.com/your-username/movieverse_fullstack.git
cd movieverse_fullstack/backend
```

 2. Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

 3. Install dependencies
```bash
pip install -r requirements.txt
```

 4. Create `.env` file
Inside the `backend/` folder, create a file named `.env` with:
```
TMDB_API_KEY=your_tmdb_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

 5. Prepare data (run once)
```bash
python scripts/preprocess.py
python scripts/build_embeddings.py
```

 6. Run the backend server
```bash
python app.py
```

Now open your browser and go to:
```
http://127.0.0.1:5000
```

---

  Deployment (Render)

 1. Push your code to GitHub
Make sure you’ve added a `.gitignore` file excluding:
```
venv/
__pycache__/
*.pkl
*.csv
.env
```

 2. Create a new Web Service on [Render](https://render.com)
- Connect your GitHub repo  
- Choose “Python” environment  
- Set the Start Command to:
  ```
  gunicorn app:app
  ```
- Add environment variables (same as `.env`):
  ```
  TMDB_API_KEY
  SECRET_KEY
  ```

Render will install your requirements and host your Flask app automatically.

---

 License
This project is for educational purposes only.
