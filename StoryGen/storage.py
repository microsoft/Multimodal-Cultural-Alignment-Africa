"""Optional Supabase Storage integration."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path


class SupabaseUploader:
    def __init__(self) -> None:
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError(
                "Supabase upload requires the supabase package"
            ) from exc

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "--upload-storage requires SUPABASE_URL and SUPABASE_KEY"
            )
        self.client = create_client(url, key)
        self.image_bucket = os.getenv("SUPABASE_IMAGE_BUCKET", "images")
        self.audio_bucket = os.getenv("SUPABASE_AUDIO_BUCKET", "audios")

    def upload(self, path: Path, object_name: str, *, media_type: str) -> str:
        bucket = self.image_bucket if media_type == "image" else self.audio_bucket
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            self.client.storage.from_(bucket).upload(
                object_name,
                handle,
                file_options={"content-type": content_type, "upsert": "true"},
            )
        return self.client.storage.from_(bucket).get_public_url(object_name)
