# Parakeet ASR example. Run targets from this directory, e.g. `make benchmark`.
# uv resolves the project venv from the repo root automatically.

PLATFORM ?= mps
MODEL ?= mlx-community/parakeet-tdt-0.6b-v2
AUDIO ?= sample.wav
ROUNDS ?= 20

.PHONY: benchmark transcribe sample clean

sample: $(AUDIO)

$(AUDIO):
	uv run --with pyarrow python fetch_sample.py

benchmark: $(AUDIO)
	JAX_PLATFORMS=$(PLATFORM) uv run python benchmark.py --model $(MODEL) --audio $(AUDIO) --rounds $(ROUNDS)

transcribe: $(AUDIO)
	JAX_PLATFORMS=$(PLATFORM) uv run python main.py --model $(MODEL) --audio $(AUDIO)

clean:
	rm -f $(AUDIO)
