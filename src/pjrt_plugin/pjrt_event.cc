// PJRT Event API implementation for Metal backend

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Event API
// ============================================================================

PJRT_Error* MPS_Event_Destroy(PJRT_Event_Destroy_Args* args) {
    delete args->event;
    return nullptr;
}

PJRT_Error* MPS_Event_IsReady(PJRT_Event_IsReady_Args* args) {
    args->is_ready = args->event ? args->event->ready : true;
    return nullptr;
}

PJRT_Error* MPS_Event_Error(PJRT_Event_Error_Args* args) {
    // Return the event's error status (nullptr means no error)
    return nullptr;
}

PJRT_Error* MPS_Event_Await(PJRT_Event_Await_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Event_OnReady(PJRT_Event_OnReady_Args* args) {
    // Call callback immediately since we're synchronous
    if (args->callback) {
        args->callback(nullptr, args->user_arg);
    }
    return nullptr;
}
