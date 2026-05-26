import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Page settings
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="centered"
)


# Title
st.title("🎵 Music Recommendation System")
st.write("Get similar song recommendations using Machine Learning")


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
        return []

    index = matches.index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:6]

    recommended = []

    for item in scores:
        recommended.append(
            songs.iloc[item[0]]["title"]
        )

    return recommended


# Dropdown
song_name = st.selectbox(
    "Choose a Song",
    songs["title"].values
)


# Button
if st.button("Recommend Songs"):

    recommendations = recommend(song_name)

    st.subheader("Recommended Songs")

    for song in recommendations:
        st.success(song)


# Footer
st.write("---")
st.caption("Built using TF-IDF and Cosine Similarity")
