#pragma once

// Records executable dispatches while a jax.profiler trace is active, for the
// PJRT plugin profiler (pjrt_profiler.cc) to export as a device plane.

#include <cstdint>
#include <string>
#include <vector>

namespace jax_mps {

// One MlxExecutable::Execute() call, in wall-clock nanoseconds since the Unix
// epoch (the clock XLA's host tracer uses, so planes line up in the viewer).
struct ExecutionTrace {
    std::string name;      // module name, e.g. "jit_f"
    int64_t start_ns;      // Execute() entered
    int64_t graph_end_ns;  // MLX graph built (or compiled function called)
    int64_t end_ns;        // outputs evaluated (or enqueued, if async)
    bool async_dispatch;   // end_ns marks enqueue, not GPU completion
};

// Current wall-clock time in ns since the epoch.
int64_t WallClockNs();

// True while a trace is being recorded. A relaxed atomic load, so Execute()
// can check it on every call at negligible cost.
bool TraceRecordingActive();

// Appends one execution to the active recording (ignored when inactive).
void RecordExecution(ExecutionTrace trace);

// Starts a new recording, discarding anything left from a previous one.
void StartTraceRecording();

// Stops recording; the events stay available to TakeTraceRecording().
void StopTraceRecording();

// Returns and clears the recorded executions.
std::vector<ExecutionTrace> TakeTraceRecording();

}  // namespace jax_mps
