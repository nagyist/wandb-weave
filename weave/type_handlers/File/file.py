"""Defines the custom File weave type."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

from typing_extensions import deprecated

from weave.trace.serialization import serializer
from weave.trace.serialization.custom_objs import MemTraceFilesArtifact

logger = logging.getLogger(__name__)

DEPRECATION_MESSAGE = """
File is deprecated and scheduled to be removed in the next release\n
Use the new Content API which supports all File methods instead:\n
File(path) -> Content.from_path(path)\n
File(path, mimetype) -> Content.from_path(path, mimetype)\n
"""


@deprecated(DEPRECATION_MESSAGE)
class File:
    """A class representing a file with path, mimetype, and size information."""

    def __init__(self, path: str | Path, mimetype: str | None = None):
        """Initialize a File object.

        Args:
            path: Path to the file (string or pathlib.Path)
            mimetype: Optional MIME type of the file - will be inferred from extension if not provided
        """
        self.path = Path(path) if isinstance(path, str) else path
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist")
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path} is not a file")
        self.mimetype = (
            mimetype if mimetype else mimetypes.guess_type(str(self.path))[0]
        )
        self.size = self.path.stat().st_size

    @property
    def filename(self) -> str:
        """Get the filename of the file.

        Returns:
            str: The name of the file without the directory path.
        """
        return self.path.name

    def open(self) -> bool:
        """Open the file using the operating system's default application.

        This method uses the platform-specific mechanism to open the file with
        the default application associated with the file's type.

        Returns:
            bool: True if the file was successfully opened, False otherwise.
        """
        try:
            if sys.platform == "win32":
                os.startfile(self.path)
            elif sys.platform == "darwin":  # macOS
                subprocess.call(("open", str(self.path)))
            else:  # linux variants
                subprocess.call(("xdg-open", str(self.path)))
        except Exception as e:
            logger.exception("Failed to open file %s", self.path)
            return False
        return True

    def save(self, dest: str | Path) -> None:
        """Copy the file to the specified destination path.

        Args:
            dest: Destination path where the file will be copied to (string or pathlib.Path)
                  The destination path can be a file or a directory.
        """
        path = Path(dest) if isinstance(dest, str) else dest
        os.makedirs(path.parent, exist_ok=True)
        shutil.copy2(self.path, path)


@deprecated(DEPRECATION_MESSAGE)
def save(obj: File, artifact: MemTraceFilesArtifact, name: str) -> None:
    from weave.type_handlers.Content.content import Content
    from weave.type_handlers.Content.content import save as save_content

    logger.warning("Saving File as Content object")
    content: Content = Content.from_path(obj.path, mimetype=obj.mimetype)
    save_content(content, artifact, name)


def load(artifact: MemTraceFilesArtifact, name: str) -> File:
    metadata_path = artifact.path("metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    path = artifact.path("file", metadata.get("original_path"))
    return File(path, metadata["mimetype"])


def register() -> None:
    serializer.register_serializer(File, save, load)
