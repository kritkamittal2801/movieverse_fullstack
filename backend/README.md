MovieVerse — Full-Stack Movie Recommendation Platform

A full-stack movie  website built using Flask, SentenceTransformers, Hugging Face datasets,  
and a custom HTML/CSS/JS frontend. Users can search movies semantically, watch video, browse details,  
and the system tracks user activity authenticated via a lightweight SQLite + HuggingFace Hub backend.

---

 Features

Movie Search & Recommendations  
- Semantic search using SentenceTransformers  
- Movie metadata + embeddings loaded from a Hugging Face dataset  
- TMDB images automatically fetched   

 Machine Learning  
- Preprocessed embeddings  
- Similarity-based recommendations  
- Optimized pickle-based data loading  

 Backend (Flask API)  
- Search and recommendation endpoints  
- User login/signup (SQLite)  
- User activity tracking (synced with HuggingFace repo)  
- Fully deployable on Render  

 Frontend  
- Custom HTML pages served via Flask templates  
- Interactive UI built with JavaScript  
- Movie browsing UI with posters, details, and video previews  
- Responsive CSS design  

---

 Project Structure


 Project Structure
```
movieverse_fullstack/
├── backend/
│ ├── app.py
│ ├── search_api.py
│ ├── fetch_movie_images_ids.py
│ ├── database.py
│ ├── static/
│ │ ├── moviewebsite.css
│ │ ├── moviewebsite.js
│ │ ├── loginDes.css
│ │ ├── loginEffects.js
│ │ ├── videoDes.css
│ │ ├── videoEffects.js
│ │ └── img/
│ ├── templates/
│ │ ├── index.html
│ │ ├── moviewebsite.html
│ │ ├── video.html
│ │ └── payment_page.html
│ ├── data/
│ ├── scripts/
│ ├── requirements.txt
│ └── README.md
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
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
USERDATA_REPO=YOUR_HF_USERDATA_REPO
TMDB_API_KEY=YOUR_TMDB_API_KEY
SECRET_KEY=YOUR_FLASK_SECRET_KEY
MOVIES_URL=URL_TO_HUGGINGFACE_MOVIES_PKL

```

 5. Fetch TMDB images
 python fetch_movie_images_ids.py
```

 6. Run the backend server
```bash
python app.py
```

Now open your browser and go to:
```
http://127.0.0.1:5000
```
The frontend is automatically served by Flask.
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
  HF_TOKEN
  USERDATA_REPO
  TMDB_API_KEY
  SECRET_KEY
  MOVIES_URL
  ```

Render will install your requirements and host your Flask app automatically.

---

 License
This project is for educational purposes only.
