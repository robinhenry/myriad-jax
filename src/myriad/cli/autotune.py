"""Auto-tune CLI commands."""

import click


@click.group()
def autotune() -> None:
    """Manage auto-tune profiles and cache."""
    pass


@autotune.command("clear-cache")
def clear_cache() -> None:
    """Delete the auto-tune profile cache (~/.myriad/autotune_profiles.json)."""
    from myriad.platform.autotune.cache import get_cache_path

    cache_path = get_cache_path()
    if not cache_path.exists():
        click.echo("No auto-tune cache found.")
        return

    cache_path.unlink()
    click.echo(f"Cleared auto-tune cache: {cache_path}")
