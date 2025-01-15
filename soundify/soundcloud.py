from typing import Optional, List, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from pydantic import BaseModel, Field


class User(BaseModel):
    full_name: str
    username: str


class PublisherMetadata(BaseModel):
    id: Optional[int] = None
    artist: Optional[str] = None
    release_title: Optional[str] = None
    album_title: Optional[str] = None
    isrc: Optional[str] = None
    explicit: Optional[bool] = None
    writer_composer: Optional[str] = None
    publisher: Optional[str] = None


class Track(BaseModel):
    id: int
    artwork_url: Optional[str] = None
    title: str
    description: Optional[str] = None
    duration: Optional[int] = Field(..., alias="full_duration")
    user: User
    label_name: Optional[str] = None
    publisher_metadata: Optional[PublisherMetadata] = None
    release_date: Optional[str] = None


class Like(BaseModel):
    track: Track


class LikesResponse(BaseModel):
    collection: List[Like]
    next_href: Optional[str] = None
    query_urn: Optional[str] = None


class SoundCloudClient:
    def __init__(self, client_id: str, user_id: int):
        self.client_id = client_id
        self.user_id = user_id
        self.base_url = "https://api-v2.soundcloud.com"
        self.default_params = {
            "client_id": self.client_id,
            "app_version": "1736508062",
            "app_locale": "en",
        }
        self._client = httpx.AsyncClient()

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _add_default_params(self, url: str) -> str:
        """Add default parameters to any URL."""
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        for key, value in self.default_params.items():
            params[key] = [value]

        new_query = urlencode(params, doseq=True)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"

    async def get_likes(
        self, next_href: Optional[str] = None, limit: int = 24
    ) -> Tuple[List[Like], Optional[str]]:
        """
        Fetch a single page of likes.

        Args:
            next_href: URL for the next page (if None, fetches first page)
            limit: Number of items per page

        Returns:
            Tuple containing:
            - List of Like objects for the current page
            - URL for the next page (None if no more pages)
        """
        if next_href:
            url = self._add_default_params(next_href)
            response = await self._client.get(url)
        else:
            url = f"{self.base_url}/users/{self.user_id}/track_likes"
            params = {
                **self.default_params,
                "limit": limit,
                "linked_partitioning": 1,
            }
            response = await self._client.get(url, params=params)

        response.raise_for_status()
        likes_response = LikesResponse.model_validate(response.json())

        return likes_response.collection, likes_response.next_href
