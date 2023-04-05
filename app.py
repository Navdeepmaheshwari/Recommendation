from flask import Flask,flash, render_template, request,redirect,send_file,jsonify
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('model.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl','rb'))
new_data = pd.read_parquet('movies.parquet')
# print(new_data)
app = Flask(__name__)
@app.route('/')
def home():
    return 'Movie Recommendation API '

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    try:
        movie = request.json['movie']
        if not movie:
            return jsonify({'error': 'Please provide a movie name.'}), 400
        indices = new_data['title'][(new_data['title'].str.contains(movie))|(new_data['title'].str.lower().str.contains(movie))].index
        if len(indices) == 0:
            similar = []
            return {'recommendations': similar}
        input = vectors.toarray()[indices[0]]
        distances_movies, indices_movies = model.kneighbors([input])
        lst = []
        for indices in indices_movies:
            lst.extend(indices)
        similar = []
        for i in lst:
            similar.append((new_data.iloc[i].title,str(new_data.iloc[i].id)))
        return jsonify({'recommendations': similar})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# recommend_movies('Spider-Man')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

