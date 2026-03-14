// Minimal PJRT profiler extension — returns errors for all operations.
// Prevents jaxlib from crashing on null function pointer dereference.

#include "pjrt_plugin/pjrt_profiler.h"

#include <cstdlib>
#include <cstring>

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
// Profiler stub type
// ============================================================================

struct PLUGIN_Profiler {};

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
// Profiler operation stubs — all return "not supported"
// ============================================================================

static PLUGIN_Profiler_Error* ProfilerCreate(PLUGIN_Profiler_Create_Args* args) {
    return MakeProfilerError("MPS profiler not supported");
}

static PLUGIN_Profiler_Error* ProfilerDestroy(PLUGIN_Profiler_Destroy_Args* args) {
    return nullptr;
}

static PLUGIN_Profiler_Error* ProfilerStart(PLUGIN_Profiler_Start_Args* args) {
    return MakeProfilerError("MPS profiler not supported");
}

static PLUGIN_Profiler_Error* ProfilerStop(PLUGIN_Profiler_Stop_Args* args) {
    return MakeProfilerError("MPS profiler not supported");
}

static PLUGIN_Profiler_Error* ProfilerCollectData(PLUGIN_Profiler_CollectData_Args* args) {
    return MakeProfilerError("MPS profiler not supported");
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
