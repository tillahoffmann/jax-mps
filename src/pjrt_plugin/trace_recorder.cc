#include "pjrt_plugin/trace_recorder.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <utility>

namespace jax_mps {

namespace {

struct Recorder {
    std::atomic<bool> active{false};
    std::mutex mutex;
    std::vector<ExecutionTrace> events;  // guarded by mutex
};

Recorder& GetRecorder() {
    static Recorder recorder;
    return recorder;
}

}  // namespace

int64_t WallClockNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

bool TraceRecordingActive() {
    return GetRecorder().active.load(std::memory_order_relaxed);
}

void RecordExecution(ExecutionTrace trace) {
    auto& recorder = GetRecorder();
    std::scoped_lock lock(recorder.mutex);
    // Re-check under the lock so an Execute() racing StopTraceRecording()
    // cannot append after the recording was collected.
    if (recorder.active.load(std::memory_order_relaxed))
        recorder.events.push_back(std::move(trace));
}

void StartTraceRecording() {
    auto& recorder = GetRecorder();
    std::scoped_lock lock(recorder.mutex);
    recorder.events.clear();
    recorder.active.store(true, std::memory_order_relaxed);
}

void StopTraceRecording() {
    auto& recorder = GetRecorder();
    std::scoped_lock lock(recorder.mutex);
    recorder.active.store(false, std::memory_order_relaxed);
}

std::vector<ExecutionTrace> TakeTraceRecording() {
    auto& recorder = GetRecorder();
    std::scoped_lock lock(recorder.mutex);
    return std::exchange(recorder.events, {});
}

}  // namespace jax_mps
