import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

movies_url = os.getenv("MOVIES_URL")
print("MOVIES_URL:", movies_url)

try:
    df = pd.read_pickle(movies_url)
    print("✅ Successfully loaded movies data!")
    print(df.head())
except Exception as e:
    print("❌ Failed to load dataset:", e)
