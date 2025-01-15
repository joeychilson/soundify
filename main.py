import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from pydantic import Field
from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

from soundify.soundcloud import SoundCloudClient
from soundify.soundify import Soundify, SoundifyConfig
from soundify.spotify import SpotifyClient

app = typer.Typer(help="Soundify - Transfer SoundCloud likes to Spotify playlists")
console = Console()


class SoundifySettings(BaseSettings):
    soundcloud_client_id: str = Field(..., description="SoundCloud Client ID")
    soundcloud_user_id: str = Field(..., description="SoundCloud User ID")
    spotify_client_id: str = Field(..., description="Spotify Client ID")
    spotify_client_secret: str = Field(..., description="Spotify Client Secret")
    spotify_redirect_uri: str = Field(..., description="Spotify Redirect URI")
    openai_api_key: str = Field(..., description="OpenAI API Key")
    batch_size: int = Field(
        default=50, description="Number of tracks to process in each batch"
    )
    search_candidates: int = Field(
        default=5, description="Number of search candidates per track"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="SOUNDIFY_",
    )


def setup_logging(verbose: bool):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("soundify")


def load_config(config_path: Optional[Path]) -> SoundifySettings:
    """Load configuration from file and environment"""
    config_data = {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            file_config = yaml.safe_load(f)
            config_data.update(file_config)

    try:
        if config_data:
            return SoundifySettings(**config_data)
        return SoundifySettings()
    except Exception as e:
        raise typer.BadParameter(f"Configuration error: {str(e)}")


@app.command()
def sync(
    limit: int = typer.Option(
        None, help="Number of liked tracks to process (default: all)"
    ),
    config_file: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Sync SoundCloud likes to a Spotify playlist"""
    logger = setup_logging(verbose)

    try:
        settings = load_config(config_file)

        soundcloud_client = SoundCloudClient(
            client_id=settings.soundcloud_client_id, user_id=settings.soundcloud_user_id
        )

        spotify_client = SpotifyClient(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
        )

        openai_client = OpenAI(api_key=settings.openai_api_key)

        soundify = Soundify(
            soundcloud_client=soundcloud_client,
            spotify_client=spotify_client,
            openai_client=openai_client,
            config=SoundifyConfig(
                batch_size=settings.batch_size,
                search_candidates=settings.search_candidates,
            ),
        )

        async def run_sync():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Processing likes...", total=None)
                result = await soundify.process_likes_to_playlist(limit=limit)

                console.print(f"\n✨ Created playlist: {result.playlist_url}")
                console.print(f"📊 Total tracks processed: {result.total_tracks}")

        asyncio.run(run_sync())

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=verbose)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
