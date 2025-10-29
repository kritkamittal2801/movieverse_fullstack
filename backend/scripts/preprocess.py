# preprocess.py
import pandas as pd
import re
import os

IN = "./data/movies_raw.csv"
OUT = "./data/movies_clean.csv"

def clean_text(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    df = pd.read_csv(IN)
    df["clean_description"] = df["description"].fillna("").apply(clean_text)
    df["genre"] = df["genre"].fillna("").apply(clean_text)

    # Extract year from release_date
    df["release_year"] = df["release_date"].astype(str).str[:4]

    df = df[df["clean_description"].str.len() > 0].reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print("Saved", OUT, "rows:", len(df))


if __name__ == "__main__":
    main()
