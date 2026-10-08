"""Computer-vision layer: camera access and per-frame detectors."""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_model(path: Path, url: str) -> Path:
    """Return a local model file, downloading it once if it is missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading %s ...", path.name)
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path
