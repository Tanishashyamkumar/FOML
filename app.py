from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables to store preloaded data
movies_data = None
similarity = None

def load_and_process_data():
    global movies_data, similarity
    
    # Load movies data
    movies_data = pd.read_csv('movies.csv')
    
    # Select and prepare features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

    # Generate feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Compute similarity matrix
    similarity = cosine_similarity(feature_vectors)

# Call the data loading function at startup
load_and_process_data()

# Home page route
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/model-details')
def model():
    return render_template('model_details.html')

@app.route('/about-us')
def about():
    return render_template('about-us.html')
# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie-input']
    list_of_all_titles = movies_data['title'].tolist()
    url = movies_data['homepage'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return render_template('result.html', movie_name=movie_name,recommendations=["No matching movies found."])


    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Get similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Get top 30 similar movies
    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[1:31], start=1):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append(f"{i}. {title_from_index}")

    return render_template('result.html', movie_name=close_match, recommendations=recommendations, url=url)

if __name__ == '__main__':
    app.run(debug=True)
