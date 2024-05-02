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

    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_var'] = np.var(spectral_centroid)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_var'] = np.var(spectral_rolloff)
    features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
    features['zero_crossing_rate_var'] = np.var(zero_crossing_rate)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['percussive_mean'] = np.mean(percussive)
    features['percussive_var'] = np.var(percussive)
    features['tempo'] = tempo

    for i in range(20):
        features[f'mfccs_{i + 1}_mean'] = np.mean(mfccs[i])
        features[f'mfccs_{i + 1}_var'] = np.var(mfccs[i])

    return features




def process_data():
    info = pd.read_csv(path_to_info)
    count = 0

    columns = [
                  'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
                  'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean',
                  'spectral_bandwidth_var', 'spectral_rolloff_mean', 'spectral_rolloff_var',
                  'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'harmony_mean', 'harmony_var',
                  'percussive_mean', 'percussive_var', 'tempo'
              ] + [f'mfccs_{i + 1}_mean' for i in range(20)] + [f'mfccs_{i + 1}_var' for i in range(20)]

    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(path_to_data) for file in files if
                 file.endswith('.mp3')]

    print(f"Found {len(all_files)} files")

    results = []
    for file in tqdm.tqdm(all_files, desc='Extracting features'):
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
    df.to_csv('data/extracted_features.csv')
    print("Features extracted and saved to 'data/extracted_features.csv'")

    return df


if __name__ == '__main__':
    process_data()
