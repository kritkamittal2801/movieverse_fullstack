import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import os

IN = "./data/movies_clean.csv"
OUT = "./data/movies_with_embeddings.pkl"

def main():
    df = pd.read_csv(IN)
    model = SentenceTransformer('all-MiniLM-L6-v2')


    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
        # Combine important fields into one string
        text = (
            str(row["title"]) + ". " +
            "Genre: " + str(row.get("genre", "")) + ". " +
            "Released in " + str(row.get("release_year", "")) + ". " +
            str(row["clean_description"])
        )
        emb = model.encode(text, convert_to_numpy=True)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    df["embedding"] = embeddings

    with open(OUT, "wb") as f:
        pickle.dump(df, f)

    print(f" Saved embeddings to {OUT}, rows: {len(df)}")


if __name__ == "__main__":
    main()
