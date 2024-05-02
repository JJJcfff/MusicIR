import os
import tqdm
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv


def auth():
    load_dotenv()
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
    ))
    return sp


def get_spotify_recommendations(seed_tracks, num_tracks=20):
    sp = auth()
    related_tracks = []
    for track_id in tqdm.tqdm(seed_tracks, desc='Getting recommendations'):
        results = sp.recommendations(seed_tracks=[track_id], limit=num_tracks)
        related_tracks.extend([(track['id'], track['name']) for track in results['tracks']])
    for track in related_tracks:
        if not download_preview(track[0]):
            related_tracks.remove(track)

    print(f'Found {len(related_tracks)} related tracks')
    return related_tracks


def download_preview(track_id, output_dir='data/spotify_previews'):
    sp = auth()
    track = sp.track(track_id)
    preview_url = track['preview_url']
    if preview_url is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    preview_file = os.path.join(output_dir, f'{track_id}.mp3')

    response = requests.get(preview_url)
    if response.status_code == 200:
        with open(preview_file, 'wb') as file:
            file.write(response.content)
        return True
    else:
        return False


get_spotify_recommendations(['2dYV68n5uCXD5nM8BWQ7kp'])
