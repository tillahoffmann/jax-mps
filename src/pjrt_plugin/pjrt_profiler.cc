// PJRT profiler extension: exports MPS executable dispatches recorded while a
// jax.profiler trace is active (see trace_recorder.h) as a "/device:GPU:0"
// XPlane, which XLA's plugin tracer merges into the trace (TensorBoard,
// Perfetto). Each execution appears on three lines: the whole Execute() call,
// the MLX graph build, and the evaluation. Timestamps are host wall-clock, so
// with JAX_MPS_ASYNC_DISPATCH the evaluation span ends at enqueue rather than
// at GPU completion (the "async_dispatch" stat marks those events).

#include "pjrt_plugin/pjrt_profiler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "pjrt_plugin/trace_recorder.h"
#include "xplane.pb.h"

// ============================================================================
// Profiler error type
// ============================================================================

struct PLUGIN_Profiler_Error {
    char message[256];
    int code;
};

static PLUGIN_Profiler_Error* MakeProfilerError(const char* msg) {
    auto* err = new PLUGIN_Profiler_Error();
    strncpy(err->message, msg, sizeof(err->message) - 1);
    err->message[sizeof(err->message) - 1] = '\0';
    err->code = 1;  // UNIMPLEMENTED
    return err;
}

// ============================================================================
// Profiler state
// ============================================================================

struct PLUGIN_Profiler {
    // Serialized XSpace from the last collect_data; XLA reads it through the
    // pointer we hand back, so it must outlive that call.
    std::string serialized;
    bool collected = false;
};

namespace {

constexpr int64_t kPsPerNs = 1000;

std::string SerializeExecutions(const std::vector<jax_mps::ExecutionTrace>& executions) {
    tensorflow::profiler::XSpace space;
    if (executions.empty())
        return space.SerializeAsString();

    auto* plane = space.add_planes();
    plane->set_id(0);
    plane->set_name("/device:GPU:0");

    // One event metadata entry per distinct executable name.
    std::map<std::string, int64_t> metadata_ids;
    for (const auto& e : executions) {
        auto [it, inserted] =
            metadata_ids.emplace(e.name, static_cast<int64_t>(metadata_ids.size()) + 1);
        if (inserted) {
            auto& md = (*plane->mutable_event_metadata())[it->second];
            md.set_id(it->second);
            md.set_name(e.name);
        }
    }
    constexpr int64_t kAsyncStatId = 1;
    auto& async_md = (*plane->mutable_stat_metadata())[kAsyncStatId];
    async_md.set_id(kAsyncStatId);
    async_md.set_name("async_dispatch");

    int64_t base_ns = executions.front().start_ns;
    for (const auto& e : executions)
        base_ns = std::min(base_ns, e.start_ns);

    struct LineSpec {
        const char* name;
        int64_t jax_mps::ExecutionTrace::* begin;
        int64_t jax_mps::ExecutionTrace::* end;
    };
    const LineSpec specs[] = {
        {"MPS executions", &jax_mps::ExecutionTrace::start_ns, &jax_mps::ExecutionTrace::end_ns},
        {"Graph build", &jax_mps::ExecutionTrace::start_ns, &jax_mps::ExecutionTrace::graph_end_ns},
        {"Evaluation", &jax_mps::ExecutionTrace::graph_end_ns, &jax_mps::ExecutionTrace::end_ns},
    };
    int64_t line_id = 0;
    for (const auto& spec : specs) {
        auto* line = plane->add_lines();
        line->set_id(line_id);
        line->set_display_id(line_id);
        ++line_id;
        line->set_name(spec.name);
        line->set_timestamp_ns(base_ns);
        for (const auto& e : executions) {
            auto* event = line->add_events();
            event->set_metadata_id(metadata_ids.at(e.name));
            event->set_offset_ps((e.*spec.begin - base_ns) * kPsPerNs);
            event->set_duration_ps(std::max<int64_t>(e.*spec.end - e.*spec.begin, 0) * kPsPerNs);
            auto* stat = event->add_stats();
            stat->set_metadata_id(kAsyncStatId);
            stat->set_int64_value(e.async_dispatch ? 1 : 0);
        }
    }
    return space.SerializeAsString();
}

}  // namespace

// ============================================================================
// Error handling stubs
// ============================================================================

static void ProfilerErrorDestroy(PLUGIN_Profiler_Error_Destroy_Args* args) {
    delete args->error;
}

static void ProfilerErrorMessage(PLUGIN_Profiler_Error_Message_Args* args) {
    if (args->error) {
        args->message = args->error->message;
        args->message_size = strlen(args->error->message);
    } else {
        args->message = "";
        args->message_size = 0;
    }
}

static PLUGIN_Profiler_Error* ProfilerErrorGetCode(PLUGIN_Profiler_Error_GetCode_Args* args) {
    if (args->error) {
        args->code = args->error->code;
    }
    return nullptr;
}

// ============================================================================
// Profiler operations
// ============================================================================

static PLUGIN_Profiler_Error* ProfilerCreate(PLUGIN_Profiler_Create_Args* args) {
    args->profiler = new PLUGIN_Profiler();
    return nullptr;
}

static PLUGIN_Profiler_Error* ProfilerDestroy(PLUGIN_Profiler_Destroy_Args* args) {
    delete args->profiler;
    return nullptr;
}

static PLUGIN_Profiler_Error* ProfilerStart(PLUGIN_Profiler_Start_Args* args) {
    if (args->profiler == nullptr)
        return MakeProfilerError("MPS profiler: start called without a profiler");
    args->profiler->collected = false;
    args->profiler->serialized.clear();
    jax_mps::StartTraceRecording();
    return nullptr;
}

static PLUGIN_Profiler_Error* ProfilerStop(PLUGIN_Profiler_Stop_Args* args) {
    jax_mps::StopTraceRecording();
    return nullptr;
}

// The C API documents a two-call protocol (size query with buffer == nullptr,
// then a copy into the caller's buffer), but XLA's PluginTracer calls once and
// parses whatever `buffer` points at. Support both: always expose our own
// buffer on the first call, and copy into a caller-provided one.
static PLUGIN_Profiler_Error* ProfilerCollectData(PLUGIN_Profiler_CollectData_Args* args) {
    auto* profiler = args->profiler;
    if (profiler == nullptr)
        return MakeProfilerError("MPS profiler: collect_data called without a profiler");
    if (!profiler->collected) {
        jax_mps::StopTraceRecording();
        profiler->serialized = SerializeExecutions(jax_mps::TakeTraceRecording());
        profiler->collected = true;
    }
    auto* data = reinterpret_cast<uint8_t*>(profiler->serialized.data());
    if (args->buffer == nullptr) {
        args->buffer = data;
    } else if (args->buffer != data) {
        std::memcpy(args->buffer, data, profiler->serialized.size());
    }
    args->buffer_size_in_bytes = profiler->serialized.size();
    return nullptr;
}

// ============================================================================
// Extension registration
// ============================================================================

static PLUGIN_Profiler_Api profiler_api = {
    .struct_size = sizeof(PLUGIN_Profiler_Api),
    .priv = nullptr,
    .error_destroy = ProfilerErrorDestroy,
    .error_message = ProfilerErrorMessage,
    .error_get_code = ProfilerErrorGetCode,
    .create = ProfilerCreate,
    .destroy = ProfilerDestroy,
    .start = ProfilerStart,
    .stop = ProfilerStop,
    .collect_data = ProfilerCollectData,
};

static PJRT_Profiler_Extension profiler_extension = {
    .base =
        {
            .struct_size = sizeof(PJRT_Extension_Base),
            .type = PJRT_Extension_Type_Profiler,
            .next = nullptr,
        },
    .profiler_api = &profiler_api,
    .traceme_context_id = 0,
};

PJRT_Extension_Base* GetProfilerExtension() {
    return &profiler_extension.base;
}
