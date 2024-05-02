from sklearn.neighbors import NearestNeighbors
import pandas as pd
from extract_feature import extract_feature
from get_spotify_recomendation import get_spotify_recommendations
def find_similar_songs(input_audio_path, features_csv_path, n_neighbors=5):
    features_df = pd.read_csv(features_csv_path, index_col='track_id')
    #exclude spotify_id and genre columns
    feature_cols = [col for col in features_df.columns if col not in ['spotify_id', 'genre']]
    
    input_features = extract_feature(input_audio_path)
    input_features_df = pd.DataFrame([input_features], columns=feature_cols)


    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(features_df[feature_cols])
    # print(features_df[feature_cols].head())
    distances, indices = nbrs.kneighbors(input_features_df)
    #flatten
    indices = indices.flatten()
    distances = distances.flatten()
    # remove index if distance = 0
    indices = indices[distances != 0]
    distances = distances[distances != 0]

    similar_songs = features_df.iloc[indices]['spotify_id'].to_dict()

    return similar_songs, distances

similar_songs, distances = find_similar_songs('data/MP3-Example/Blues/Blues-TRACOHF128F1498509.mp3', 'data/extracted_features.csv')
print("Similar Songs:", similar_songs)
print("Distances:", distances)
