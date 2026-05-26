import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# Load dataset
songs = pd.read_csv("songs.csv")


# Combine features
songs["features"] = (
    songs["genre"] + " " +
    songs["mood"] + " " +
    songs["artist"]
)


# Convert text into vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(songs["features"])


# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)


# Recommendation function
def recommend(song_name):

    matches = songs[songs["title"] == song_name]

    if matches.empty:
        print("\nSong not found!")
        return

    index = matches.index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:6]

    print("\nRecommended Songs:\n")

    for item in scores:
        print(songs.iloc[item[0]]["title"])


# Show songs
print("\nChoose Song Number:\n")

for i, title in enumerate(songs["title"].head(20)):
    print(i, "-", title)


# User choice
choice = int(input("\nEnter number: "))

song = songs["title"][choice]


# Get recommendations
recommend(song)
