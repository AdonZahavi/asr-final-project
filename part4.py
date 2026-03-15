from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from common_voice import DEFAULT_CLIPS_DIR, DEFAULT_MUSAN_DIR, DEFAULT_NOISY_DIR, DEFAULT_TEST_TSV, iter_test_samples


MIN_SNR = 0.0
MAX_SNR = 6.0
DEFAULT_LOG_FILE = Path("augmentation_log.tsv")


def get_valid_noise_files(musan_dir: Path | str) -> list[Path]:
    musan_dir = Path(musan_dir)
    valid_files: list[Path] = []
    for noise_path in musan_dir.glob("noise/free-sound/**/*.wav"):
        try:
            if sf.info(noise_path).duration >= 30:
                valid_files.append(noise_path)
        except RuntimeError:
            continue
    return valid_files


def calculate_power(signal_data: np.ndarray) -> float:
    return float(np.mean(np.square(signal_data)))


def ensure_mono(signal_data: np.ndarray) -> np.ndarray:
    if signal_data.ndim == 1:
        return signal_data
    return np.mean(signal_data, axis=1)


def maybe_downsample(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate > 16000:
        audio = signal.decimate(audio, q=2, axis=0)
        sample_rate = sample_rate // 2
    return audio, sample_rate


def add_noise_to_speech(speech: np.ndarray, noise: np.ndarray, target_snr: float) -> np.ndarray:
    speech_power = calculate_power(speech)
    noise_power = calculate_power(noise)
    if noise_power == 0.0:
        return speech

    target_ratio = 10 ** (target_snr / 10.0)
    alpha = math.sqrt(speech_power / (target_ratio * noise_power))
    mixed = speech + alpha * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def build_noisy_test_set(
    test_tsv: Path | str = DEFAULT_TEST_TSV,
    clips_dir: Path | str = DEFAULT_CLIPS_DIR,
    musan_dir: Path | str = DEFAULT_MUSAN_DIR,
    output_dir: Path | str = DEFAULT_NOISY_DIR,
    log_file: Path | str = DEFAULT_LOG_FILE,
    seed: int = 13,
) -> None:
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_file)
    noise_files = get_valid_noise_files(musan_dir)

    if not noise_files:
        raise FileNotFoundError(f"No valid MUSAN noise files found under {musan_dir}")

    samples = list(iter_test_samples(test_tsv=test_tsv, clips_dir=clips_dir))
    with log_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Filename", "Background file", "Start point (in seconds)", "SNR"],
            delimiter="\t",
        )
        writer.writeheader()

        for sample in samples:
            speech_path = Path(sample["audio_path"])
            if not speech_path.exists():
                print(f"Missing audio file: {speech_path}")
                continue

            try:
                speech, sample_rate = sf.read(speech_path)
            except RuntimeError as exc:
                print(f"Error reading {speech_path.name}: {exc}")
                continue

            speech = ensure_mono(np.asarray(speech, dtype=np.float32))
            speech, sample_rate = maybe_downsample(speech, sample_rate)

            noise_path = random.choice(noise_files)
            noise_info = sf.info(noise_path)
            speech_len = len(speech)
            max_start = noise_info.frames - speech_len
            if max_start <= 0:
                continue

            start_sample = random.randint(0, max_start)
            noise, _ = sf.read(noise_path, start=start_sample, frames=speech_len)
            noise = ensure_mono(np.asarray(noise, dtype=np.float32))
            current_snr = random.uniform(MIN_SNR, MAX_SNR)
            noisy_audio = add_noise_to_speech(speech, noise, current_snr)

            output_path = output_dir / f"{sample['filename']}.wav"
            sf.write(output_path, noisy_audio, sample_rate)
            writer.writerow(
                {
                    "Filename": sample["filename"],
                    "Background file": noise_path.name,
                    "Start point (in seconds)": f"{start_sample / noise_info.samplerate:.2f}",
                    "SNR": f"{current_snr:.2f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part D: add MUSAN noise to the Common Voice Hebrew test split.")
    parser.add_argument("--test-tsv", default=str(DEFAULT_TEST_TSV), help="Path to Common Voice test.tsv.")
    parser.add_argument("--clips-dir", default=str(DEFAULT_CLIPS_DIR), help="Path to original clips directory.")
    parser.add_argument("--musan-dir", default=str(DEFAULT_MUSAN_DIR), help="Path to extracted MUSAN dataset.")
    parser.add_argument("--output-dir", default=str(DEFAULT_NOISY_DIR), help="Directory for noisy audio files.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE), help="Augmentation log TSV.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible augmentation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_noisy_test_set(
        test_tsv=args.test_tsv,
        clips_dir=args.clips_dir,
        musan_dir=args.musan_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        seed=args.seed,
    )
