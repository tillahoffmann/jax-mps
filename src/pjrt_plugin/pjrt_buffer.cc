// PJRT Buffer API implementation for Metal backend

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Buffer API
// ============================================================================

PJRT_Error* MPS_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args) {
    delete args->buffer;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args) {
    args->type = args->buffer && args->buffer->buffer
                     ? static_cast<PJRT_Buffer_Type>(args->buffer->buffer->dtype())
                     : PJRT_Buffer_Type_F32;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        const auto& dims = args->buffer->buffer->dimensions();
        args->unpadded_dims = dims.data();
        args->num_dims = dims.size();
    } else {
        args->unpadded_dims = nullptr;
        args->num_dims = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_DynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    args->layout.type = PJRT_Buffer_MemoryLayout_Type_Strides;
    return nullptr;
}

PJRT_Error* MPS_Buffer_OnDeviceSizeInBytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    args->on_device_size_in_bytes =
        args->buffer && args->buffer->buffer ? args->buffer->buffer->byte_size() : 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Buffer_Device called, buffer=%p\n", (void*)args->buffer);
    if (args->buffer && args->buffer->client && !args->buffer->client->devices.empty()) {
        args->device = args->buffer->client->devices[0];
        MPS_LOG_DEBUG(" PJRT_Buffer_Device: returning device=%p from client=%p\n",
                      (void*)args->device, (void*)args->buffer->client);
    } else {
        args->device = nullptr;
        MPS_LOG_DEBUG(" PJRT_Buffer_Device: returning nullptr (no client or devices)\n");
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Buffer_Memory called\n");
    // Return the default memory for the buffer's device
    if (args->buffer && args->buffer->client && !args->buffer->client->memories.empty()) {
        args->memory = args->buffer->client->memories[0];
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Delete(PJRT_Buffer_Delete_Args* args) {
    if (args->buffer && args->buffer->buffer) {
        args->buffer->buffer->Delete();
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args) {
    args->is_deleted =
        args->buffer && args->buffer->buffer ? args->buffer->buffer->IsDeleted() : true;
    return nullptr;
}

PJRT_Error* MPS_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args) {
    return MakeError("CopyToDevice not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    if (args->src && args->src->buffer && args->dst) {
        args->src->buffer->ToHostBuffer(args->dst, nullptr);
    }

    auto* event = new PJRT_Event();
    event->ready = true;
    args->event = event;

    return nullptr;
}

PJRT_Error* MPS_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args) {
    args->is_on_cpu = false;
    return nullptr;
}

PJRT_Error* MPS_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args) {
    auto* event = new PJRT_Event();
    event->ready = true;
    args->event = event;
    return nullptr;
}

PJRT_Error* MPS_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args) {
    args->buffer_pointer = 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
    args->device_memory_ptr = nullptr;
    return nullptr;
}

// ============================================================================
// CopyToDeviceStream API
// ============================================================================

PJRT_Error* MPS_CopyToDeviceStream_Destroy(PJRT_CopyToDeviceStream_Destroy_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_AddChunk(PJRT_CopyToDeviceStream_AddChunk_Args* args) {
    return MakeError("CopyToDeviceStream not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_CopyToDeviceStream_TotalBytes(PJRT_CopyToDeviceStream_TotalBytes_Args* args) {
    args->total_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_GranuleSize(PJRT_CopyToDeviceStream_GranuleSize_Args* args) {
    args->granule_size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_CopyToDeviceStream_CurrentBytes(PJRT_CopyToDeviceStream_CurrentBytes_Args* args) {
    args->current_bytes = 0;
    return nullptr;
}
