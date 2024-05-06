import librosa
import numpy as np
import pandas as pd
import os
import tqdm

path_to_data = 'data/MP3-Example/'
path_to_info = 'data/Music Info.csv'
hop_length = 5000


def extract_feature(file_name):
    y, sr = librosa.load(file_name)  # mixed to mono and resampled to 22050 Hz

    features = {}
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    harmony, percussive = librosa.effects.hpss(y)

    features['chroma_stft'] = chroma_stft
    features['rms'] = rms
    features['spectral_centroid'] = spectral_centroid
    features['spectral_bandwidth'] = spectral_bandwidth
    features['spectral_rolloff'] = spectral_rolloff
    features['zero_crossing_rate'] = zero_crossing_rate
    features['harmony'] = harmony
    features['percussive'] = percussive
    features['tempo'] = tempo

    for i in range(20):
        features[f'mfccs_{i + 1}'] = mfccs[i]

    return features


def process_data():
    info = pd.read_csv(path_to_info)
    count = 0

    columns = [
                  'chroma_stft', 'rms',
                  'spectral_centroid',
                  'spectral_bandwidth',
                  'spectral_rolloff',
                  'zero_crossing_rate',
                  'harmony',
                  'percussive',
                  'tempo'
              ] + [f'mfccs_{i + 1}' for i in range(20)]

    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_to_data) for file in files if
                 file.endswith('.mp3')]

    print(f"Found {len(all_files)} files")

    results = []
    for file in tqdm.tqdm(all_files, desc='Extracting series'):
        # if count == 10:
        #     break
        track_id = os.path.basename(file).split('-')[1].split('.')[0]
        spotify_id = info[info['track_id'] == track_id]['spotify_id'].values[0]
        genre = os.path.basename(file).split('-')[0]

        features = extract_feature(file)
        results.append([track_id, spotify_id, genre] + [features[col] for col in columns])
        count += 1

    df = pd.DataFrame(results, columns=['track_id', 'spotify_id', 'genre'] + columns)
    df.set_index('track_id', inplace=True)
    df.to_csv('data/extracted_series.csv')
    print("Features extracted and saved to 'data/extracted_series.csv'")

    return df


if __name__ == '__main__':
    process_data()
