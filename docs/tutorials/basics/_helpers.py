"""Notebook display helpers (not rendered in docs)."""

import base64
from pathlib import Path

from IPython.display import HTML


def side_by_side_videos(paths: list[Path], labels: list[str], width: int = 200) -> HTML:
    """Display videos side by side with labels."""
    cells = []
    for path, label in zip(paths, labels):
        data = base64.b64encode(path.read_bytes()).decode()
        cells.append(
            f'<div style="text-align:center">'
            f"<b>{label}</b><br>"
            f'<video width="{width}" controls loop autoplay muted>'
            f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
            f"</video></div>"
        )
    return HTML(f'<div style="display:flex;gap:20px">{"".join(cells)}</div>')
