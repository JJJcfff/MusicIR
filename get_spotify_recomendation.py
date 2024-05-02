import os
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
    for track_id in seed_tracks:
        results = sp.recommendations(seed_tracks=[track_id], limit=num_tracks)
        related_tracks.extend([(track['id'], track['name']) for track in results['tracks']])
    return related_tracks

