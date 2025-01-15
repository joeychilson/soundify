import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel

from soundify.soundcloud import Like, SoundCloudClient
from soundify.spotify import SearchResult, SpotifyClient

logger = logging.getLogger(__name__)


class PlaylistCreationResult(BaseModel):
    """Results from creating and populating a Spotify playlist."""

    playlist_id: str
    playlist_url: str
    total_tracks: int


class SoundifyConfig(BaseModel):
    """Configuration settings for Soundify."""

    batch_size: int = 50
    search_candidates: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0


class Soundify:
    """Main class for managing SoundCloud to Spotify migrations."""

    def __init__(
        self,
        soundcloud_client: SoundCloudClient,
        spotify_client: SpotifyClient,
        openai_client: OpenAI,
        config: Optional[SoundifyConfig] = None,
    ):
        self.soundcloud = soundcloud_client
        self.spotify = spotify_client
        self.openai = openai_client
        self.config = config or SoundifyConfig()

    async def get_all_likes(self, limit: Optional[int] = None) -> List[Like]:
        """
        Retrieve all likes from SoundCloud, handling pagination automatically.

        Args:
            limit: Optional maximum number of likes to retrieve
            progress_callback: Optional callback function(current_count: int, total: Optional[int])

        Returns:
            List of Like objects
        """
        likes: List[Like] = []
        next_href = None
        retry_count = 0

        while True:
            if limit and len(likes) >= limit:
                likes = likes[:limit]
                break

            try:
                remaining = limit - len(likes) if limit else self.config.batch_size
                batch_size = (
                    min(remaining, self.config.batch_size)
                    if limit
                    else self.config.batch_size
                )

                batch_likes, next_href = await self.soundcloud.get_likes(
                    next_href=next_href, limit=batch_size
                )

                if batch_likes:
                    likes.extend(batch_likes)
                    retry_count = 0

                    logger.debug(
                        f"Retrieved {len(batch_likes)} likes. Total: {len(likes)}"
                    )

                if not next_href or not batch_likes:
                    break

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Error fetching likes (attempt {retry_count}): {str(e)}"
                )

                if retry_count >= self.config.max_retries:
                    logger.error("Max retries reached. Stopping like retrieval.")
                    break

                await asyncio.sleep(self.config.retry_delay)

        logger.info(f"Retrieved {len(likes)} total likes")
        return likes

    async def _generate_search_query(self, like: Like) -> str:
        """Generate the most effective Spotify search query for a track."""
        track_data = {
            "id": like.track.id,
            "title": like.track.title,
            "description": like.track.description,
            "user": like.track.user.model_dump(),
            "publisher_metadata": like.track.publisher_metadata.model_dump()
            if like.track.publisher_metadata
            else None,
            "label_name": like.track.label_name,
            "duration": like.track.duration,
            "release_date": like.track.release_date,
        }

        prompt = """
Given a SoundCloud track's metadata, generate the most effective Spotify search query.
IMPORTANT: Only use "artist:" and "track:" in queries.

Track Analysis Criteria:

1. Check Title for Artist and Title:
- "Artist - Title" → artist:"Artist" track:"Title"
- "Artist - Title (Remix)" → artist:"Artist" track:"Title (Remix)"
- "Artist Title" → artist:"Artist" track:"Title"

Remove these suffixes from the track title:
- "(Original Mix)"
- "(Extended Mix)"
- "(Radio Edit)"
- "(Club Mix)"
- "(VIP Mix)"
- "(Dub Mix)"
- "(Edit)"
- "(Extended)"
- "(Original Version)"
- "(Club Version)"
- "(Radio Version)"

Remove any featuring/featured artists from the track title:
- "Title ft. Featured" → "Title"
- "Title feat. Featured" → "Title"
- "Title (ft. Featured)" → "Title"
- "Title (feat. Featured)" → "Title"

2. Check Description for Artist and Title: Look for patterns like the following in the description:
- "Artist released"
- "Artist will release"
- "by Artist"
- "produced by Artist"
- "Artist:"
- "from Artist"
- "Music by Artist"
- "Track by Artist"

3. Fallback Artist Criteria:
If artist still cannot be determined, use these in order:
a) publisher_metadata.artist if available
b) track.user.username if available 

4. Fallback Track Title Criteria:
If track title cannot be determined from parsed title, use these in order:
a) publisher_metadata.release_title if available
b) Original track title with cleanup

Title Cleanup Rules:
Remove from start:
- "Premiere:"
- "Premier:"
- "Free Download:"
- "#TBT"

Remove from end:
- "[Label Name]"
- "(OUT NOW)"
- "[Free Download]"

Return a JSON object with a single "query" field containing the generated search query.

Examples:

Artist from title:
{
    "query": "artist:\"Lane 8\" track:\"Brightest Lights\""
}

Artist from description:
{
    "query": "artist:\"Ben Böhmer\" track:\"Breathing\""
}

With remix kept:
{
    "query": "artist:\"Above & Beyond\" track:\"Sun In Your Eyes (Spencer Brown Remix)\""
}

Using publisher metadata artist:
{
    "query": "artist:\"Lane 8\" track:\"Brightest Lights\""
}
"""

        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a music metadata expert that generates Spotify search queries.",
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nAnalyze this track:\n{json.dumps(track_data)}",
                },
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(completion.choices[0].message.content)
        return result["query"]

    async def _validate_match(
        self, like: Like, candidates: List[SearchResult]
    ) -> Optional[str]:
        """Use OpenAI to validate and select the best matching Spotify track URI."""
        validation_prompt = f"""
Compare this SoundCloud track with potential Spotify matches and identify the best match.

Consider:
1. Title similarity
2. Artist name matches
3. Album name matches
4. Label name matches
5. Release year if available

Favor tracks that are:
- Most similar in data
- More popular
- Apart of albums, eps, or singles
- Not apart of Radio shows, or compilations.

SoundCloud Track:
{json.dumps(like.model_dump(), indent=2)}

Spotify Candidates:
{json.dumps([c.model_dump() for c in candidates], indent=2)}

Return a JSON object with a single "spotify_uri" field containing either the URI of the best match or null if no good match is found.

Example responses:

For a good match:
{{
    "spotify_uri": "spotify:track:1234567890abcdef"
}}

For no match:
{{
    "spotify_uri": null
}}
    """

        completion = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a music expert that validates track matches.",
                },
                {"role": "user", "content": validation_prompt},
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(completion.choices[0].message.content)
        return result["spotify_uri"]

    async def find_spotify_match(self, like: Like) -> Optional[str]:
        """Find the best matching Spotify track URI for a SoundCloud like."""
        try:
            if like.track.publisher_metadata and like.track.publisher_metadata.isrc:
                isrc_results = self.spotify.search_tracks(
                    query=f"isrc:{like.track.publisher_metadata.isrc}", limit=1
                )
                if isrc_results and isrc_results[0].uri:
                    return isrc_results[0].uri

            llm_query = await self._generate_search_query(like)

            logger.info(
                f"Finding Spotify match for {like.track.title} with query: {llm_query}"
            )

            primary_results = self.spotify.search_tracks(
                query=llm_query, limit=self.config.search_candidates
            )

            backup_results = self.spotify.search_tracks(
                query=like.track.title, limit=self.config.search_candidates
            )

            combined_results = []
            seen_uris = set()

            for result in primary_results + backup_results:
                if result.uri not in seen_uris:
                    combined_results.append(result)
                    seen_uris.add(result.uri)

            if combined_results:
                return await self._validate_match(like, combined_results)

            return None

        except Exception as e:
            logger.error(
                f"Error finding Spotify match for track {like.track.title}: {str(e)}"
            )
            return None

    async def process_likes(self, limit: Optional[int] = None) -> List[str]:
        """Process all likes and find Spotify matches."""
        matches = []
        likes = await self.get_all_likes(limit=limit)

        for i, like in enumerate(likes):
            logger.info(f"Processing track {i + 1} of {len(likes)}")

            uri = await self.find_spotify_match(like)
            if uri:
                matches.append(uri)
                logger.info(f"Found match for track {like.track.title}")
            else:
                logger.info(f"No match found for track {like.track.title}")

        return matches

    async def create_playlist_from_matches(
        self,
        track_uris: List[str],
        playlist_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> PlaylistCreationResult:
        """Create a Spotify playlist from matched track URIs."""
        if not playlist_name:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            playlist_name = f"SoundCloud Likes - {timestamp}"

        if not description:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            description = (
                f"Imported from SoundCloud likes on {timestamp}. "
                f"Contains {len(track_uris)} matched tracks."
            )

        logger.info(f"Creating playlist: {playlist_name}")

        playlist_id = self.spotify.get_or_create_playlist(playlist_name, description)
        self.spotify.add_tracks_to_playlist(playlist_name, track_uris, description)

        result = PlaylistCreationResult(
            playlist_id=playlist_id,
            playlist_url=f"https://open.spotify.com/playlist/{playlist_id}",
            total_tracks=len(track_uris),
        )

        logger.info(
            f"Playlist creation complete: {result.playlist_url} with {result.total_tracks} tracks"
        )
        return result

    async def process_likes_to_playlist(
        self,
        limit: Optional[int] = None,
        playlist_name: Optional[str] = None,
        playlist_description: Optional[str] = None,
    ) -> PlaylistCreationResult:
        """
        Process likes, find matches, and create a playlist in one operation.

        Args:
            limit: Optional maximum number of likes to process
            playlist_name: Optional custom playlist name
            playlist_description: Optional playlist description

        Returns:
            PlaylistCreationResult with creation statistics
        """
        track_uris = await self.process_likes(limit=limit)

        return await self.create_playlist_from_matches(
            track_uris=track_uris,
            playlist_name=playlist_name,
            description=playlist_description,
        )
