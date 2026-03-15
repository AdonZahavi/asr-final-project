from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Iterator


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = Path(os.environ.get("COMMONVOICE_HE_DIR", PROJECT_DIR / "cv-corpus-24.0-2025-12-05-he.tar" / "cv-corpus-24.0-2025-12-05" / "he"))
DEFAULT_TEST_TSV = DEFAULT_DATASET_DIR / "test.tsv"
DEFAULT_CLIPS_DIR = DEFAULT_DATASET_DIR / "clips"
DEFAULT_NOISY_DIR = PROJECT_DIR / "noisy_clips"
DEFAULT_MUSAN_DIR = Path(os.environ.get("MUSAN_DIR", PROJECT_DIR / "musan"))
DEFAULT_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"


def read_test_rows(test_tsv: Path | str = DEFAULT_TEST_TSV) -> list[Dict[str, str]]:
    test_tsv = Path(test_tsv)
    with test_tsv.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()

    if not lines:
        return []

    header = lines[0].rstrip("\n\r").split("\t")
    split_limit = max(len(header) - 1, 0)
    rows: list[Dict[str, str]] = []

    for raw_line in lines[1:]:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        parts = line.split("\t", split_limit)
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        rows.append({column: value for column, value in zip(header, parts)})

    return rows


def iter_test_samples(
    test_tsv: Path | str = DEFAULT_TEST_TSV,
    clips_dir: Path | str = DEFAULT_CLIPS_DIR,
    audio_extension: str | None = None,
) -> Iterator[Dict[str, str]]:
    clips_dir = Path(clips_dir)
    for row in read_test_rows(test_tsv):
        audio_name = (row.get("path") or "").strip()
        sentence = (row.get("sentence") or "").strip()
        if not audio_name:
            continue
        if audio_extension:
            audio_name = f"{Path(audio_name).stem}{audio_extension}"
        yield {
            "filename": Path(audio_name).stem,
            "audio_name": audio_name,
            "audio_path": str(clips_dir / audio_name),
            "reference_text": sentence,
        }


def ensure_parent_dir(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tsv_dict_reader(path: Path | str) -> Iterable[Dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.readlines()

    if not lines:
        return

    header = lines[0].rstrip("\n\r").split("\t")
    split_limit = max(len(header) - 1, 0)

    for raw_line in lines[1:]:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        parts = line.split("\t", split_limit)
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        yield {column: value for column, value in zip(header, parts)}
