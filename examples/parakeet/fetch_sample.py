"""Download the demo audio clip used by the README / benchmark.

Extracts the first utterance of the `hf-internal-testing/librispeech_asr_dummy`
dataset (LibriSpeech dev-clean 1272-128104-0000, "Mister Quilter ...") and writes
it to sample.wav next to this script. Requires pyarrow to read the parquet; run via
`uv run --with pyarrow python fetch_sample.py` (the Makefile does this for you).
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-not-found]  # ephemeral: `uv run --with pyarrow`
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "hf-internal-testing/librispeech_asr_dummy"
PARQUET = "clean/validation-00000-of-00001.parquet"
OUT = Path(__file__).parent / "sample.wav"


def main():
    path = hf_hub_download(REPO, PARQUET, repo_type="dataset")
    row = pq.read_table(path).to_pylist()[0]
    audio = row["audio"]
    data = audio["bytes"] if isinstance(audio, dict) else audio
    wav, sr = sf.read(io.BytesIO(data))
    sf.write(str(OUT), wav, sr, subtype="PCM_16")
    print(f"wrote {OUT} ({len(wav) / sr:.2f}s @ {sr} Hz)")
    print(f"reference transcript: {row['text']}")


if __name__ == "__main__":
    main()
