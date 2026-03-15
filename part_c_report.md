# ASR Final Report

## Project goal

The project evaluates the Hebrew ASR model `ivrit-ai/whisper-large-v3-turbo-ct2` on the Mozilla Common Voice Hebrew test set.

## Part A

For Part A, I transcribed the `test.tsv` split and saved the results in:

- `results_part_a.tsv`

The output format is:

- `Filename`
- `Reference Text`
- `Transcribed Text`

All text files were handled as UTF-8.

## Part B: baseline evaluation

The baseline evaluation was computed from the raw transcription output using word-level alignment and edit statistics.

- Input file: `results_part_a.tsv`
- Output file: `results_part_b.tsv`
- Alignment log: `part_b_alignment_log.tsv`
- Frequent errors report: `part_b_frequent_errors.tsv`

Final baseline results on the clean benchmark:

- WER: `0.3585`
- Recall: `0.6579`
- Precision: `0.6560`
- F1: `0.6570`

## Part C: error analysis and normalization

### Analysis method

I analyzed the model output in two complementary ways:

1. I inspected the full alignment log for individual utterances.
2. I reviewed the most frequent error pairs over the full benchmark.

This made it possible to separate true ASR errors from artifacts caused by the evaluation procedure.

### Main issues found

The raw evaluation penalized the system for many differences that do not reflect real recognition failures:

- punctuation differences
- Hebrew diacritics and quote marks
- spelling variants such as `haita` / `hayta`
- spelling variants of words such as `hakol`
- spelling variants such as `misim` / `masim`
- some number expressions written as words in one side and digits in the other

I intentionally did not normalize ambiguous pairs that may change meaning.

### Final normalization rules

The final normalization pipeline performs:

1. removal of Hebrew diacritics
2. removal of punctuation and quote-like symbols
3. whitespace normalization
4. conservative normalization of common orthographic variants
5. limited normalization of frequent number words into digits

### Final clean-benchmark results

- Input file: `results_part_a.tsv`
- Output file: `results_part_c.tsv`
- Alignment log: `part_c_alignment_log.tsv`
- Frequent errors report: `part_c_frequent_errors.tsv`

Final normalized results on the clean benchmark:

- WER: `0.0773`
- Recall: `0.9302`
- Precision: `0.9309`
- F1: `0.9305`

This reached the assignment target of single-digit WER.

## Part D: noisy audio robustness

For the noise condition, the assignment requires the `mod 6 = 0` setting:

- background type: noise
- SNR range: `0-6 dB`

Only long files from `MUSAN/noise/free-sound/` were used, as required. Each Common Voice test clip was downsampled to `16 kHz` using `scipy.signal.decimate(..., q=2)` before mixing.

Generated files:

- augmentation log: `augmentation_log.tsv`
- noisy transcription file: `results_part_a_noisy.tsv`
- noisy normalized evaluation file: `results_part_c_noisy.tsv`
- noisy alignment log: `part_c_noisy_alignment_log.tsv`
- noisy frequent errors report: `part_c_noisy_frequent_errors.tsv`

Final normalized results on the noisy benchmark:

- WER: `0.1858`
- Recall: `0.8304`
- Precision: `0.8340`
- F1: `0.8322`

### Interpretation

The model remains usable under strong background noise, but performance drops substantially relative to the clean benchmark:

- clean normalized WER: `0.0773`
- noisy normalized WER: `0.1858`

The increase in WER shows that background noise causes more substitutions, deletions, and insertions, especially function words and short words.

## Conclusion

The raw baseline significantly underestimated the real quality of the model because it penalized many formatting and orthographic differences. After targeted normalization, the clean-benchmark WER dropped from `0.3585` to `0.0773`. Under strong background noise, the normalized WER increased to `0.1858`, showing a clear but expected robustness degradation.
