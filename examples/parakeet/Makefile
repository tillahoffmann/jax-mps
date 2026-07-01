# Parakeet ASR example. Run targets from this directory, e.g. `make benchmark`.
# uv resolves the project venv from the repo root automatically.

PLATFORM ?= mps
MODEL ?= mlx-community/parakeet-tdt-0.6b-v2
AUDIO ?= sample.wav
ROUNDS ?= 20
# Public-domain JFK speech clip (16 kHz mono WAV, ~11s), from whisper.cpp.
AUDIO_URL ?= https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav

.PHONY: benchmark transcribe sample bench-async bench-batch clean

sample: $(AUDIO)

$(AUDIO):
	curl -sL -o $(AUDIO) $(AUDIO_URL)

benchmark: $(AUDIO)
	JAX_PLATFORMS=$(PLATFORM) uv run python benchmark.py --model $(MODEL) --audio $(AUDIO) --rounds $(ROUNDS)

transcribe: $(AUDIO)
	JAX_PLATFORMS=$(PLATFORM) uv run python main.py --model $(MODEL) --audio $(AUDIO)

# A/B the plugin's async dispatch (encoder is dispatch-bound; no on-device
# control flow, so it should be safe under async_eval).
bench-async: $(AUDIO)
	@echo "=== async dispatch OFF ==="
	JAX_PLATFORMS=$(PLATFORM) uv run python benchmark.py --dtype bfloat16 --rounds $(ROUNDS)
	@echo "=== async dispatch ON ==="
	JAX_MPS_ASYNC_DISPATCH=1 JAX_PLATFORMS=$(PLATFORM) uv run python benchmark.py --dtype bfloat16 --rounds $(ROUNDS)

# Encoder throughput vs batch size (amortized per-clip time).
bench-batch: $(AUDIO)
	JAX_PLATFORMS=$(PLATFORM) uv run python bench_batch.py --rounds $(ROUNDS)

clean:
	rm -f $(AUDIO)
