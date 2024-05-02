from sklearn.neighbors import NearestNeighbors
import pandas as pd
from extract_feature import extract_feature
import os
def find_similar_songs(input_audio_path, features_csv_path, n_neighbors=5):
    features_df = pd.read_csv(features_csv_path, index_col='track_id')
    
    input_features = extract_feature(input_audio_path)
    input_features_df = pd.DataFrame([input_features], columns=features_df.columns) 

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(features_df.values)  
    distances, indices = nbrs.kneighbors(input_features_df)

    similar_songs = features_df.iloc[indices[0]].index
    return similar_songs, distances

similar_songs, distances = find_similar_songs('/Users/chujian/Downloads/Data/genres_original/blues/blues.00023.wav', 'data/extracted_features.csv')
print("Similar Songs:", similar_songs)
print("Distances:", distances)
