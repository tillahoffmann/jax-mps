// PJRT Memory API implementation for Metal backend

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Memory API
// ============================================================================

PJRT_Error* MPS_Memory_Id(PJRT_Memory_Id_Args* args) {
    args->id = 0;
    return nullptr;
}

PJRT_Error* MPS_Memory_Kind(PJRT_Memory_Kind_Args* args) {
    static const char* kind_str = "device";
    args->kind = kind_str;
    args->kind_size = 6;
    return nullptr;
}

PJRT_Error* MPS_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
    static const char* str = "MPS Memory";
    args->debug_string = str;
    args->debug_string_size = 10;
    return nullptr;
}

PJRT_Error* MPS_Memory_ToString(PJRT_Memory_ToString_Args* args) {
    static const char* str = "MpsMemory()";
    args->to_string = str;
    args->to_string_size = 11;
    return nullptr;
}

PJRT_Error* MPS_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Memory_AddressableByDevices called\n");
    if (args->memory && args->memory->device) {
        static PJRT_Device* dev_array[1];
        dev_array[0] = args->memory->device;
        args->devices = dev_array;
        args->num_devices = 1;
    } else {
        args->devices = nullptr;
        args->num_devices = 0;
    }
    return nullptr;
}
