import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

movies = pd.read_csv('movies.csv')

tfidf = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf.fit_transform(movies['genres'])

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(genre_matrix)

def recommend_movies(movie_name, n=4):
    try:
        movie_idx = movies[movies['title'].str.contains(movie_name, case=False)].index[0]
        distances, indices = knn.kneighbors(genre_matrix[movie_idx], n_neighbors=n+1)
        recommendations = movies.iloc[indices[0][1:]]['title'].tolist()
        return recommendations
    except IndexError:
        return ["Movie not found."]

st.title("Movie Recommendation System")
movie_name = st.text_input("Enter a movie you enjoyed:")
if st.button("Recommend"):
    recommendations = recommend_movies(movie_name)
    st.write("Recommended movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
