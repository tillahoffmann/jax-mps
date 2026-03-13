// PJRT Executable and LoadedExecutable API implementation for Metal backend

#include <string>
#include <vector>

#include "device_assignment.pb.h"
#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_mutex.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Executable API
// ============================================================================

PJRT_Error* MPS_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
    // Skip deletion if executable is owned by a LoadedExecutable
    // (will be deleted when LoadedExecutable is destroyed)
    if (args->executable && args->executable->owned_by_loaded) {
        return nullptr;
    }
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPS_Executable_Name(PJRT_Executable_Name_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_Name called\n");
    static const char* name = "mps_executable";
    args->executable_name = name;
    args->executable_name_size = 14;
    return nullptr;
}

PJRT_Error* MPS_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args) {
    args->num_replicas = 1;
    return nullptr;
}

PJRT_Error* MPS_Executable_NumPartitions(PJRT_Executable_NumPartitions_Args* args) {
    args->num_partitions = 1;
    return nullptr;
}

PJRT_Error* MPS_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_NumOutputs called\n");
    args->num_outputs = args->executable && args->executable->executable
                            ? args->executable->executable->num_outputs()
                            : 1;
    MPS_LOG_DEBUG(" PJRT_Executable_NumOutputs: %zu\n", args->num_outputs);
    return nullptr;
}

PJRT_Error* MPS_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    args->size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    args->num_properties = 0;
    args->properties = nullptr;
    return nullptr;
}

PJRT_Error* MPS_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_OutputMemoryKinds called\n");

    if (!args->executable) {
        return MakeError("Null executable", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    // Initialize dynamic storage if needed
    args->executable->initOutputMetadata();

    size_t num_outputs = args->executable->output_memory_kinds.size();
    args->num_outputs = num_outputs;
    args->memory_kinds = args->executable->output_memory_kinds.data();
    args->memory_kind_sizes = args->executable->output_memory_kind_sizes.data();
    MPS_LOG_DEBUG(" PJRT_Executable_OutputMemoryKinds: %zu outputs\n", num_outputs);
    return nullptr;
}

PJRT_Error* MPS_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_OutputElementTypes called\n");

    if (!args->executable) {
        return MakeError("Null executable", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    // Initialize dynamic storage if needed
    args->executable->initOutputMetadata();

    size_t num_outputs = args->executable->output_types.size();
    args->output_types = args->executable->output_types.data();
    args->num_output_types = num_outputs;
    MPS_LOG_DEBUG(" PJRT_Executable_OutputElementTypes: %zu outputs\n", num_outputs);
    return nullptr;
}

PJRT_Error* MPS_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_OutputDimensions called\n");

    if (!args->executable) {
        return MakeError("Null executable", PJRT_Error_Code_INVALID_ARGUMENT);
    }

    // Initialize dynamic storage if needed
    args->executable->initOutputMetadata();

    size_t num_outputs = args->executable->output_dim_sizes.size();
    args->num_outputs = num_outputs;
    args->dims = args->executable->output_dims.data();
    args->dim_sizes = args->executable->output_dim_sizes.data();
    MPS_LOG_DEBUG(" PJRT_Executable_OutputDimensions: %zu outputs\n", num_outputs);
    return nullptr;
}

PJRT_Error* MPS_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args) {
    return MakeError("OptimizedProgram not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
    return MakeError("Serialize not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Executable_Fingerprint called\n");
    // Return a static fingerprint for the executable
    static const char* fingerprint = "mps_exec_fingerprint";
    args->executable_fingerprint = fingerprint;
    args->executable_fingerprint_size = 20;
    return nullptr;
}

// ============================================================================
// LoadedExecutable API
// ============================================================================

PJRT_Error* MPS_LoadedExecutable_Destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    if (args->executable) {
        // Delete the owned PJRT_Executable first (owns MpsExecutable with MLIR context)
        delete args->executable->executable;
    }
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_GetExecutable called, loaded_exec=%p\n",
                  (void*)args->loaded_executable);
    if (args->loaded_executable) {
        MPS_LOG_DEBUG(" Getting executable from loaded, executable=%p\n",
                      (void*)args->loaded_executable->executable);
        args->executable = args->loaded_executable->executable;
    } else {
        args->executable = nullptr;
    }
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_GetExecutable returning executable=%p\n",
                  (void*)args->executable);
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_AddressableDevices called\n");
    MPS_LOG_DEBUG("   args->executable=%p\n", (void*)args->executable);

    // Return devices from the LoadedExecutable's client
    if (args->executable && args->executable->client &&
        !args->executable->client->devices.empty()) {
        args->addressable_devices = args->executable->client->devices.data();
        args->num_addressable_devices = args->executable->client->devices.size();
        MPS_LOG_DEBUG("   Returning %zu devices\n", args->num_addressable_devices);
    } else {
        args->addressable_devices = nullptr;
        args->num_addressable_devices = 0;
        MPS_LOG_DEBUG("   Returning 0 devices\n");
    }

    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_AddressableDevices returning\n");
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_IsDeleted called\n");
    args->is_deleted = false;
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    std::scoped_lock lock(GetPjrtGlobalMutex());
    MPS_LOG_INFO("Executing program\n");

    if (!args->executable || !args->executable->executable) {
        return MakeError("No executable to execute");
    }

    auto* mlx_executable = args->executable->executable->executable.get();
    if (!mlx_executable || !mlx_executable->IsValid()) {
        std::string error = mlx_executable ? mlx_executable->error() : "null executable";
        return MakeError("Invalid executable: " + error);
    }

    // Get the client and device
    auto* client = args->executable->client;
    if (!client || !client->client) {
        return MakeError("No client for execution");
    }

    // Gather input buffers
    std::vector<jax_mps::MlxBuffer*> inputs;
    size_t num_args = args->num_args;
    MPS_LOG_DEBUG("Execute: num_args=%zu\n", num_args);

    for (size_t i = 0; i < num_args; ++i) {
        // Each argument list has args->num_args buffers
        // For single device, we just use the first device's arguments
        PJRT_Buffer* const* arg_buffers = args->argument_lists[0];
        if (arg_buffers[i] && arg_buffers[i]->buffer) {
            inputs.push_back(arg_buffers[i]->buffer.get());
        } else {
            return MakeError("Null input buffer at index " + std::to_string(i));
        }
    }

    // Execute
    auto result = mlx_executable->Execute(inputs);

    // Check for execution errors - validate output count matches expected
    size_t expected_outputs = mlx_executable->num_outputs();
    if (result.buffers.size() != expected_outputs) {
        return MakeError("Output count mismatch: expected " + std::to_string(expected_outputs) +
                         ", got " + std::to_string(result.buffers.size()));
    }

    // Allocate output buffer array
    size_t num_outputs = result.buffers.size();
    MPS_LOG_DEBUG("Execute: num_outputs=%zu\n", num_outputs);

    // For single device execution, we have 1 device and num_outputs buffers
    // args->output_lists[0] is pre-allocated array of PJRT_Buffer* with num_outputs entries
    PJRT_Buffer** output_list = args->output_lists[0];

    for (size_t i = 0; i < num_outputs; ++i) {
        // Create new PJRT_Buffer wrapping the MlxBuffer
        auto* pjrt_buffer = new PJRT_Buffer();
        pjrt_buffer->buffer = std::move(result.buffers[i]);
        pjrt_buffer->client = client;
        output_list[i] = pjrt_buffer;
    }

    // Set the ready event (execution is synchronous for now)
    if (args->device_complete_events) {
        args->device_complete_events[0] = new PJRT_Event{.ready = true};
    }

    MPS_LOG_INFO("Execution complete\n");
    return nullptr;
}

PJRT_Error* MPS_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    return MakeError("GetCompiledMemoryStats not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_GetCompileOptions(PJRT_Executable_GetCompileOptions_Args* args) {
    return MakeError("GetCompileOptions not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
    return MakeError("DeserializeAndLoad not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_Fingerprint called\n");
    args->executable_fingerprint = nullptr;
    args->executable_fingerprint_size = 0;
    return nullptr;
}

// Backing storage for device assignment serialization
struct MpsDeviceAssignmentSerialized {
    std::string data;
};

static void MpsDeviceAssignmentDeleter(PJRT_DeviceAssignmentSerialized* da) {
    MPS_LOG_DEBUG(" MpsDeviceAssignmentDeleter called\n");
    delete reinterpret_cast<MpsDeviceAssignmentSerialized*>(da);
}

PJRT_Error* MPS_LoadedExecutable_GetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_GetDeviceAssignment called\n");

    // Create DeviceAssignment proto: 1 replica, 1 computation, device 0
    xla::DeviceAssignmentProto proto;
    proto.set_replica_count(1);
    proto.set_computation_count(1);
    auto* comp_device = proto.add_computation_devices();
    comp_device->add_replica_device_ids(0);

    auto* serialized = new MpsDeviceAssignmentSerialized();
    proto.SerializeToString(&serialized->data);

    args->serialized_bytes = serialized->data.data();
    args->serialized_bytes_size = serialized->data.size();
    args->serialized_device_assignment =
        reinterpret_cast<PJRT_DeviceAssignmentSerialized*>(serialized);
    args->serialized_device_assignment_deleter = MpsDeviceAssignmentDeleter;

    MPS_LOG_DEBUG(" PJRT_LoadedExecutable_GetDeviceAssignment returning %zu bytes\n",
                  args->serialized_bytes_size);
    return nullptr;
}
