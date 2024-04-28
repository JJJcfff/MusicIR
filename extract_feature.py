import librosa
import numpy as np
import pandas as pd
import os
import tqdm

path_to_data = 'data/MP3-Example/'
hop_length = 512


def extract_feature(file_name):
    y, sr = librosa.load(file_name)  # mixed to mono and resampled to 22050 Hz

    features = {}
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)

    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

    for i in range(20):
        features[f'mfccs_{i + 1}_mean'] = np.mean(mfccs[i])
        features[f'mfccs_{i + 1}_var'] = np.var(mfccs[i])

    return features


def process_data():
    columns = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
               'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var'] + \
              [f'mfccs_{i + 1}_mean' for i in range(20)] + [f'mfccs_{i + 1}_var' for i in range(20)]

    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_to_data) for file in files if
                 file.endswith('.mp3')]

    print(f"Found {len(all_files)} files")

    results = []
    for file in tqdm.tqdm(all_files, desc='Extracting features'):
        track_id = os.path.basename(file).split('-')[1].split('.')[0]
        features = extract_feature(file)
        results.append([track_id] + [features[col] for col in columns])

    df = pd.DataFrame(results, columns=['track_id'] + columns)
    df.set_index('track_id', inplace=True)
    df.to_csv('data/extracted_features.csv')
    print("Features extracted and saved to 'data/extracted_features.csv'")

    return df


process_data()
