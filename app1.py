#python -m uvicorn app1:app1 --host 0.0.0.0 --port 10000
import pandas as pd
import uvicorn
import json
import pickle
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from memory_profiler import memory_usage
app1 = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open('model.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl','rb'))
new_data = pd.read_parquet('movies.parquet')

@app1.get('/')
def home():
    return 'Movie Recommendation API '

@app1.post('/recommend')
async def recommend_movies(request:Request):
    try:
        
        movi= await request.json()
        movie = movi.get('movie')
        # movie='batman'
        # print(movie)
        if not movie:
            return {'error': 'Please provide a movie name.'}, 400
        indices = new_data['title'][(new_data['title'].str.contains(movie))|(new_data['title'].str.lower().str.contains(movie))].index
        if len(indices) == 0:
            similar = []
            return {'recommendations': similar}
        input = vectors[indices[0]]
        distances_movies, indices_movies = model.kneighbors(input)
        lst = []
        for indices in indices_movies:
            lst.extend(indices)
        similar = []
        for i in lst:
            similar.append({'title': new_data.iloc[i].title, 'id': str(new_data.iloc[i].id)})
        return {'recommendations': similar}
    except Exception as e:
        return {'error': str(e)}, 500
    
# if __name__ == '__main__':
#     mem_usage = max(memory_usage(proc=recommend_movies))
#     print(f"Maximum memory usage: {mem_usage} MiB")
