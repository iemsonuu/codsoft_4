import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = {
    "Movie": [
        "The Matrix", "Inception", "Interstellar", "The Dark Knight", "The Social Network",
        "Avatar", "Titanic", "Gladiator", "The Godfather", "Pulp Fiction",
        "Drishyam", "Premam", "Bangalore Days", "Kumbalangi Nights", "Maheshinte Prathikaram",
        "Lucifer", "Charlie", "Uyare", "C U Soon", "Joji",
        "Baahubali: The Beginning", "Baahubali: The Conclusion", "Dangal", "PK", "Lagaan",
        "3 Idiots", "Swades", "Zindagi Na Milegi Dobara", "Tamasha", "Barfi!",
    ],
    "Genres": [
        "Action, Sci-Fi", "Action, Adventure, Sci-Fi", "Adventure, Drama, Sci-Fi", 
        "Action, Crime, Drama", "Biography, Drama, History", "Action, Adventure, Fantasy",
        "Drama, Romance", "Action, Adventure, Drama", "Crime, Drama", "Crime, Drama, Thriller",
        "Crime, Drama, Mystery", "Romance, Drama, Comedy", "Drama, Romance, Comedy", 
        "Drama, Family, Comedy", "Drama, Comedy",
        "Action, Drama, Thriller", "Drama, Romance", "Drama", 
        "Thriller, Drama, Mystery", "Crime, Drama, Thriller",
        "Action, Adventure, Fantasy", "Action, Adventure, Fantasy", "Drama, Biography, Sport", 
        "Comedy, Drama, Sci-Fi", "Drama, History, Sport", 
        "Comedy, Drama", "Drama, Adventure", "Drama, Comedy, Adventure", 
        "Drama, Romance", "Comedy, Drama, Romance",
    ],
    "Description": [
        "A computer hacker discovers a shocking truth about reality.", 
        "A thief steals corporate secrets through dream-sharing technology.", 
        "A team travels through a wormhole to ensure humanity's survival.", 
        "A vigilante battles crime in a corrupt city.", 
        "The founding of Facebook by Harvard students.", 
        "A paraplegic marine explores an alien planet.", 
        "A love story set aboard the ill-fated Titanic.", 
        "A Roman general seeks revenge after betrayal.", 
        "A crime saga of an Italian-American family.", 
        "Interconnected stories of crime and redemption.",
        "A man resorts to deceit to save his family from a crime charge.", 
        "A journey of love across different phases of life.", 
        "A story of friendship and relationships in Bangalore.", 
        "The life of four brothers in a coastal village.", 
        "A man sets out to reclaim his dignity after a public insult.", 
        "The rise of a political leader amidst treachery.", 
        "A carefree man transforms lives with his adventures.", 
        "A young woman overcomes challenges after a traumatic event.", 
        "A gripping story told through a computer screen.", 
        "A man’s descent into crime and its consequences.",
        "A prince fights to reclaim his kingdom.", 
        "A battle to save a kingdom and avenge betrayal.", 
        "A wrestler overcomes odds to achieve greatness.", 
        "A man's quest to meet God through science and faith.", 
        "Villagers unite to challenge British rule through cricket.",
        "Three friends navigate life and its challenges.", 
        "A NASA scientist returns to his roots in India.", 
        "Three friends embark on a life-changing road trip.", 
        "An artist discovers himself through love and adventure.", 
        "A differently-abled man finds love and meaning in life.",
    ],
    "Ratings": [
        8.7, 8.8, 8.6, 9.0, 7.7, 7.8, 7.9, 8.5, 9.2, 8.9, 
        8.5, 8.3, 8.3, 8.6, 8.4, 7.5, 7.7, 8.0, 7.9, 8.1,
        8.0, 8.2, 8.4, 8.1, 8.0, 8.4, 8.2, 8.3, 7.4, 8.1,
    ],
}

df = pd.DataFrame(data)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["Genres"] + " " + df["Description"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

ratings_array = np.array(df["Ratings"]).reshape(-1, 1)
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(ratings_array)

def content_recommendations(title, cosine_sim=cosine_sim, df=df):
    idx = df[df["Movie"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["Movie"].iloc[movie_indices]

def collaborative_recommendations(title, knn=knn, df=df):
    idx = df[df["Movie"] == title].index[0]
    distances, indices = knn.kneighbors([[df["Ratings"].iloc[idx]]], n_neighbors=6)
    movie_indices = indices.flatten()[1:]
    return df["Movie"].iloc[movie_indices]

def hybrid_recommendations(title, df=df):
    content_recs = content_recommendations(title)
    collab_recs = collaborative_recommendations(title)
    combined_recs = pd.concat([content_recs, collab_recs]).drop_duplicates().head(5)
    return combined_recs

if __name__ == "__main__":
    print("Available Movies:")
    print(df["Movie"].to_string(index=False))

    movie_title = input("\nEnter a movie title to get recommendations: ")

    if movie_title in df["Movie"].values:
        print("\nContent-Based Recommendations:")
        print(content_recommendations(movie_title).to_string(index=False))

        print("\nCollaborative Recommendations:")
        print(collaborative_recommendations(movie_title).to_string(index=False))

        print("\nHybrid Recommendations:")
        print(hybrid_recommendations(movie_title).to_string(index=False))
    else:
        print("Movie not found in the dataset. Please try again.")
