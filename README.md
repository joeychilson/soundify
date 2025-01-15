# Soundify

Soundify is a tool that helps you migrate your SoundCloud likes to a Spotify playlist with a high degree of accuracy.

## Note

This is a work in progress.

## Usage

```bash
uv run main.py --limit 100 --verbose --config config.yaml
```

## Environment Variables

```
SOUNDIFY_SOUNDCLOUD_CLIENT_ID=""
SOUNDIFY_SOUNDCLOUD_USER_ID=""
SOUNDIFY_SPOTIFY_CLIENT_ID=""
SOUNDIFY_SPOTIFY_CLIENT_SECRET=""
SOUNDIFY_SPOTIFY_REDIRECT_URI=""
SOUNDIFY_OPENAI_API_KEY=""
SOUNDIFY_BATCH_SIZE=50
SOUNDIFY_SEARCH_CANDIDATES=5
```

## Config

```yaml
soundcloud_client_id: xxx
soundcloud_user_id: xxx
spotify_client_id: xxx
spotify_client_secret: xxx
spotify_redirect_uri: xxx
openai_api_key: xxx
batch_size: 50
search_candidates: 5
```
