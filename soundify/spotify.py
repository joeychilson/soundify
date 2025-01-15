from typing import List, Optional

from pydantic import BaseModel, HttpUrl
import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SearchResult(BaseModel):
    artists: List[str]
    title: str
    album: str
    popularity: int
    isrc: Optional[str]
    explicit: bool
    artwork_url: Optional[HttpUrl]
    uri: str
    duration: int
    label: Optional[str]
    release_date: Optional[str]


class SpotifyClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
    ):
        self.spotify = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="playlist-modify-public playlist-modify-private playlist-read-private",
            )
        )

    def search_tracks(
        self,
        query: str,
        limit: int = 20,
        market: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for tracks on Spotify and return structured results.

        Args:
            query: Search query string
            limit: Maximum number of results (default: 20, max: 50)
            market: ISO 3166-1 alpha-2 country code for market-specific results

        Returns:
            List of SearchResult objects containing track information
        """
        limit = min(50, max(1, limit))

        results = self.spotify.search(
            q=query,
            type="track",
            limit=limit,
            market=market,
        )

        search_results = []

        for track in results["tracks"]["items"]:
            artist_names = [artist["name"] for artist in track["artists"]]
            album = track["album"]

            artwork_url = None
            if album["images"]:
                artwork_url = album["images"][0]["url"]

            search_results.append(
                SearchResult(
                    artists=artist_names,
                    title=track["name"],
                    album=album["name"],
                    popularity=track["popularity"],
                    isrc=track["external_ids"].get("isrc"),
                    explicit=track["explicit"],
                    artwork_url=artwork_url,
                    uri=track["uri"],
                    duration=track["duration_ms"],
                    label=album.get("label"),
                    release_date=album.get("release_date"),
                )
            )

        return search_results

    def get_or_create_playlist(
        self, name: str, description: Optional[str] = None
    ) -> str:
        """
        Get existing playlist ID or create new playlist if it doesn't exist.

        Args:
            name: Name of the playlist
            description: Optional playlist description

        Returns:
            Playlist ID
        """
        user_id = self.spotify.current_user()["id"]

        playlists = self.spotify.user_playlists(user_id)
        while playlists:
            for playlist in playlists["items"]:
                if playlist["name"] == name:
                    return playlist["id"]
            playlists = self.spotify.next(playlists) if playlists["next"] else None

        playlist = self.spotify.user_playlist_create(
            user_id, name, public=True, description=description
        )
        return playlist["id"]

    def add_tracks_to_playlist(
        self,
        playlist_name: str,
        track_uris: List[str],
        description: Optional[str] = None,
    ) -> None:
        """
        Add tracks to a playlist, creating it if it doesn't exist.

        Args:
            playlist_name: Name of the playlist
            track_uris: List of Spotify track URIs to add
            description: Optional playlist description
        """
        playlist_id = self.get_or_create_playlist(playlist_name, description)

        for i in range(0, len(track_uris), 100):
            batch = track_uris[i : i + 100]
            self.spotify.playlist_add_items(playlist_id, batch)
