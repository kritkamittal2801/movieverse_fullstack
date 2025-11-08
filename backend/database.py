from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import threading
import requests
from huggingface_hub import HfApi

# SQLite database (local file)
DATABASE_URL = "sqlite:///movieverse_users.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# User table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# User activity table
class UserActivity(Base):
    __tablename__ = "user_activity"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    movie=Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Create tables

def init_db():
    Base.metadata.create_all(bind=engine)

# ---------- Hugging Face sync helpers ----------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERDATA_REPO = os.getenv("USERDATA_REPO")  
_hf_lock = threading.Lock()  

def download_db_from_hf(local_path="movieverse_users.db"):
    """Download movieverse_users.db from HF repo if present. Safe to call at startup."""
    if not HF_TOKEN or not HF_USERDATA_REPO:
        print("HF_TOKEN or USERDATA_REPO not set — skipping HF DB restore.")
        return

    url = f"https://huggingface.co/datasets/{HF_USERDATA_REPO}/resolve/main/{local_path}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        with _hf_lock:
            print(" Checking Hugging Face for existing DB...")
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200 and r.content:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(" Restored database from Hugging Face.")
            else:
                print(" No DB found on Hugging Face (status: {}).".format(r.status_code))
    except Exception as e:
        print(" Error downloading DB from Hugging Face:", e)

def upload_db_to_hf(local_path="movieverse_users.db"):
    """Upload local DB file to Hugging Face repo (overwrites). Call after commits."""
    if not HF_TOKEN or not HF_USERDATA_REPO:
        print("HF_TOKEN or USERDATA_REPO not set — skipping HF DB upload.")
        return

    if not os.path.exists(local_path):
        print("⚠️ Local DB file not found, nothing to upload.")
        return

    api = HfApi()
    try:
        with _hf_lock:
            print(f" Uploading {local_path} to Hugging Face repo {HF_USERDATA_REPO} ...")
            # NOTE: upload_file will create/replace the file in the repo root
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=os.path.basename(local_path),
                repo_id=HF_USERDATA_REPO,
                repo_type="dataset",  # dataset repo is fine; change if you used a model repo
                token=HF_TOKEN
            )
            print(" Upload to Hugging Face complete.")
    except Exception as e:
        print(" Error uploading DB to Hugging Face:", e)

