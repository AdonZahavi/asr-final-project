from __future__ import annotations

import argparse
import csv
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

from common_voice import DEFAULT_CLIPS_DIR, DEFAULT_MODEL_ID, DEFAULT_TEST_TSV, ensure_parent_dir, iter_test_samples


DEFAULT_OUTPUT_FILE = Path("results_part_a.tsv")


def get_processed_files(output_file: Path | str) -> set[str]:
    output_file = Path(output_file)
    processed: set[str] = set()
    if not output_file.exists():
        return processed

    with output_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            filename = (row.get("Filename") or "").strip()
            if filename:
                processed.add(filename)
    return processed


def transcribe_audio(model: WhisperModel, audio_path: str) -> str:
    segments, _ = model.transcribe(audio_path, language="he", beam_size=5)
    return " ".join(segment.text.strip() for segment in segments if segment.text).strip()


def run_transcription_resumable(
    test_tsv: Path | str = DEFAULT_TEST_TSV,
    clips_dir: Path | str = DEFAULT_CLIPS_DIR,
    output_file: Path | str = DEFAULT_OUTPUT_FILE,
    model_id: str = DEFAULT_MODEL_ID,
    audio_extension: str | None = None,
    device: str = "auto",
    compute_type: str = "int8",
) -> None:
    output_file = ensure_parent_dir(output_file)
    processed_files = get_processed_files(output_file)
    samples = list(iter_test_samples(test_tsv=test_tsv, clips_dir=clips_dir, audio_extension=audio_extension))

    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    write_header = not output_file.exists()

    with output_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Filename", "Reference Text", "Transcribed Text"],
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()

        for sample in tqdm(samples, unit="clip"):
            if sample["filename"] in processed_files:
                continue

            audio_path = Path(sample["audio_path"])
            if not audio_path.exists():
                print(f"Missing audio file: {audio_path}")
                continue

            try:
                transcription = transcribe_audio(model, str(audio_path))
            except Exception as exc:
                print(f"Error processing {audio_path.name}: {exc}")
                continue

            writer.writerow(
                {
                    "Filename": sample["filename"],
                    "Reference Text": sample["reference_text"],
                    "Transcribed Text": transcription,
                }
            )
            handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part A: transcribe Common Voice Hebrew test split.")
    parser.add_argument("--test-tsv", default=str(DEFAULT_TEST_TSV), help="Path to Common Voice test.tsv.")
    parser.add_argument("--clips-dir", default=str(DEFAULT_CLIPS_DIR), help="Directory that contains the audio files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE), help="Output TSV path.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Whisper model identifier.")
    parser.add_argument("--device", default="auto", help="Model device, for example auto or cpu.")
    parser.add_argument("--compute-type", default="int8", help="Model compute type, for example int8.")
    parser.add_argument(
        "--audio-extension",
        default=None,
        help="Optional replacement extension for audio files, for example .wav for noisy clips.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_transcription_resumable(
        test_tsv=args.test_tsv,
        clips_dir=args.clips_dir,
        output_file=args.output,
        model_id=args.model_id,
        audio_extension=args.audio_extension,
        device=args.device,
        compute_type=args.compute_type,
    )
