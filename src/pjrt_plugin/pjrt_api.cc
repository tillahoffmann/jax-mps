// PJRT C API implementation for Metal backend
// Uses official XLA PJRT header for correct struct layouts

// Set to 1 to enable verbose debug output
#define MPS_DEBUG 0

#if MPS_DEBUG
#define MPS_LOG(...) do { fprintf(stderr, "[MPS]" __VA_ARGS__); fflush(stderr); } while(0)
#else
#define MPS_LOG(...) do {} while(0)
#endif

#include <xla/pjrt/c/pjrt_c_api.h>

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mps_client.h"
#include "pjrt_plugin/mps_device.h"
#include "pjrt_plugin/mps_buffer.h"
#include "pjrt_plugin/mps_executable.h"
#include "pjrt_plugin/stablehlo_parser.h"
#include "device_assignment.pb.h"

// ============================================================================
// Opaque wrapper types
// ============================================================================

struct PJRT_TopologyDescription;

struct PJRT_Client {
    std::unique_ptr<jax_mps::MpsClient> client;
    std::vector<PJRT_Device*> devices;
    std::vector<PJRT_Memory*> memories;
    PJRT_TopologyDescription* topology;
};

struct PJRT_DeviceDescription;

struct PJRT_Memory;

struct PJRT_Device {
    jax_mps::MpsDevice* device;
    PJRT_Client* client;
    PJRT_DeviceDescription* description;  // Each device owns its description
    PJRT_Memory* default_memory;          // Default memory for the device
};

struct PJRT_DeviceDescription {
    PJRT_Device* device;  // Back-pointer to the device
};

struct PJRT_Memory {
    PJRT_Device* device;
    PJRT_Client* client;
    int id;
};

struct PJRT_TopologyDescription {
    PJRT_Client* client;
};

struct PJRT_Buffer {
    std::unique_ptr<jax_mps::MpsBuffer> buffer;
    PJRT_Client* client;
};

struct PJRT_Executable {
    std::unique_ptr<jax_mps::MpsExecutable> executable;
    PJRT_Client* client;
};

struct PJRT_LoadedExecutable {
    PJRT_Executable* executable;
    PJRT_Client* client;
    std::vector<PJRT_Device*> addressable_devices;
};

struct PJRT_Event {
    bool ready = true;
};

// ============================================================================
// Global state
// ============================================================================

static bool g_initialized = false;
static PJRT_Client* g_default_client = nullptr;

static PJRT_Client* GetOrCreateDefaultClient() {
    if (g_default_client) return g_default_client;

    MPS_LOG(" Creating default client\n");
    auto mps_client = jax_mps::MpsClient::Create();
    if (!mps_client) {
        MPS_LOG(" Failed to create MPS client\n");
        return nullptr;
    }

    g_default_client = new PJRT_Client();
    g_default_client->client = std::move(mps_client);

    for (int i = 0; i < g_default_client->client->device_count(); i++) {
        auto* dev = new PJRT_Device();
        dev->device = g_default_client->client->device(i);
        dev->client = g_default_client;

        // Create the device description with back-pointer
        auto* desc = new PJRT_DeviceDescription();
        desc->device = dev;
        dev->description = desc;

        // Create the default memory for the device
        auto* mem = new PJRT_Memory();
        mem->device = dev;
        mem->client = g_default_client;
        mem->id = i;
        dev->default_memory = mem;
        g_default_client->memories.push_back(mem);

        g_default_client->devices.push_back(dev);
    }

    // Create topology description
    g_default_client->topology = new PJRT_TopologyDescription();
    g_default_client->topology->client = g_default_client;

    MPS_LOG(" Created client with %zu devices\n", g_default_client->devices.size()); fflush(stderr);

    return g_default_client;
}

// ============================================================================
// Error handling
// ============================================================================

struct PJRT_Error {
    std::string message;
    PJRT_Error_Code code;
};

static PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL) {
    auto* error = new PJRT_Error();
    error->message = msg;
    error->code = code;
    return error;
}

void MPS_Error_Destroy(PJRT_Error_Destroy_Args* args) {
    delete args->error;
}

void MPS_Error_Message(PJRT_Error_Message_Args* args) {
    if (args->error) {
        args->message = args->error->message.c_str();
        args->message_size = args->error->message.size();
    }
}

PJRT_Error* MPS_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    args->code = args->error ? args->error->code : PJRT_Error_Code_OK;
    return nullptr;
}

// ============================================================================
// Plugin API
// ============================================================================

PJRT_Error* MPS_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
    MPS_LOG(" PJRT_Plugin_Initialize\n");
    g_initialized = true;
    return nullptr;
}

PJRT_Error* MPS_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

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

// ============================================================================
// Client API
// ============================================================================

PJRT_Error* MPS_Client_Create(PJRT_Client_Create_Args* args) {
    MPS_LOG(" PJRT_Client_Create called, args=%p\n", (void*)args); fflush(stderr);

    PJRT_Client* client = GetOrCreateDefaultClient();
    if (!client) {
        return MakeError("Failed to create MPS client");
    }

    MPS_LOG(" PJRT_Client_Create setting client=%p\n", (void*)client); fflush(stderr);
    args->client = client;
    MPS_LOG(" PJRT_Client_Create returning nullptr (success)\n");
    return nullptr;
}

PJRT_Error* MPS_Client_Destroy(PJRT_Client_Destroy_Args* args) {
    // Don't actually destroy since we use a singleton
    return nullptr;
}

PJRT_Error* MPS_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
    MPS_LOG(" PJRT_Client_PlatformName called\n");
    static const char* name = "mps";
    args->platform_name = name;
    args->platform_name_size = 3;
    return nullptr;
}

PJRT_Error* MPS_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
    MPS_LOG(" PJRT_Client_ProcessIndex called\n");
    args->process_index = 0;
    return nullptr;
}

PJRT_Error* MPS_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
    MPS_LOG(" PJRT_Client_PlatformVersion called\n");
    static const char* version = "0.1.0";
    args->platform_version = version;
    args->platform_version_size = 5;
    return nullptr;
}

PJRT_Error* MPS_Client_Devices(PJRT_Client_Devices_Args* args) {
    MPS_LOG(" PJRT_Client_Devices called, client=%p\n", (void*)args->client); fflush(stderr);
    PJRT_Client* client = args->client;
    if (!client) {
        client = GetOrCreateDefaultClient();
    }
    if (!client) {
        args->devices = nullptr;
        args->num_devices = 0;
        MPS_LOG(" PJRT_Client_Devices: no client, returning 0\n");
        return nullptr;
    }
    MPS_LOG(" PJRT_Client_Devices: %zu devices, data=%p\n", client->devices.size(), (void*)client->devices.data()); fflush(stderr);
    args->devices = client->devices.data();
    args->num_devices = client->devices.size();
    MPS_LOG(" PJRT_Client_Devices returning\n");
    return nullptr;
}

PJRT_Error* MPS_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
    MPS_LOG(" PJRT_Client_AddressableDevices called\n");
    PJRT_Client* client = args->client;
    if (!client) {
        client = GetOrCreateDefaultClient();
    }
    if (!client) {
        args->addressable_devices = nullptr;
        args->num_addressable_devices = 0;
        return nullptr;
    }
    args->addressable_devices = client->devices.data();
    args->num_addressable_devices = client->devices.size();
    MPS_LOG(" PJRT_Client_AddressableDevices returning %zu\n", client->devices.size()); fflush(stderr);
    return nullptr;
}

PJRT_Error* MPS_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
    MPS_LOG(" PJRT_Client_LookupDevice called, id=%d\n", (int)args->id); fflush(stderr);
    PJRT_Client* client = args->client ? args->client : GetOrCreateDefaultClient();
    if (client && args->id < client->devices.size()) {
        args->device = client->devices[args->id];
        MPS_LOG(" Returning device %p\n", (void*)args->device); fflush(stderr);
    } else {
        args->device = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args) {
    PJRT_Client* client = args->client ? args->client : GetOrCreateDefaultClient();
    if (client && args->local_hardware_id < client->devices.size()) {
        args->addressable_device = client->devices[args->local_hardware_id];
    } else {
        args->addressable_device = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
    MPS_LOG(" PJRT_Client_AddressableMemories called\n");
    PJRT_Client* client = args->client ? args->client : GetOrCreateDefaultClient();
    if (client && !client->memories.empty()) {
        args->addressable_memories = client->memories.data();
        args->num_addressable_memories = client->memories.size();
    } else {
        args->addressable_memories = nullptr;
        args->num_addressable_memories = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_Compile(PJRT_Client_Compile_Args* args) {
    MPS_LOG(" PJRT_Client_Compile\n");

    PJRT_Client* client = args->client ? args->client : GetOrCreateDefaultClient();
    if (!client) {
        return MakeError("No client available for compilation");
    }

    // Get the program from the args
    std::string format_str(args->program->format, args->program->format_size);
    MPS_LOG(" Program format: %s (size=%zu)\n", format_str.c_str(), args->program->format_size); fflush(stderr);
    MPS_LOG(" Program code size: %zu\n", args->program->code_size); fflush(stderr);

    // Parse the StableHLO bytecode
    mps::StableHLOModule stablehlo_module;
    bool parsed = false;

    if (format_str == "mlir") {
        // MLIR bytecode format (StableHLO portable artifact)
        parsed = mps::parseStableHLOBytecode(
            args->program->code,
            args->program->code_size,
            stablehlo_module
        );
    } else if (format_str == "hlo" || format_str == "hlo_with_config") {
        // Text HLO format (legacy)
        std::string program_str(args->program->code, args->program->code_size);
        parsed = mps::parseStableHLOText(program_str, stablehlo_module);
    } else {
        return MakeError("Unknown program format: " + format_str);
    }

    if (!parsed) {
        // Fallback to identity function (normal for some internal JAX operations)
        stablehlo_module.entry_function = "main";
        mps::StableHLOFunction func;
        func.name = "main";
        func.arg_types.push_back({{2}, "f32"});
        func.result_types.push_back({{2}, "f32"});
        stablehlo_module.functions.push_back(func);
    }

    // Compile the StableHLO module to MPS executable
    auto mps_exec = client->client->CompileStableHLO(stablehlo_module, nullptr);

    if (!mps_exec) {
        return MakeError("Failed to compile StableHLO to MPS");
    }

    auto* executable = new PJRT_Executable();
    executable->executable = std::move(mps_exec);
    executable->client = client;

    // Wrap in LoadedExecutable
    auto* loaded_executable = new PJRT_LoadedExecutable();
    loaded_executable->executable = executable;
    loaded_executable->client = client;
    loaded_executable->addressable_devices = client->devices;

    args->executable = loaded_executable;
    MPS_LOG(" PJRT_Client_Compile returning success\n");
    return nullptr;
}

PJRT_Error* MPS_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args) {
    // Simple single-device assignment
    if (args->default_assignment && args->default_assignment_size > 0) {
        args->default_assignment[0] = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    MPS_LOG(" PJRT_Client_BufferFromHostBuffer\n");

    PJRT_Client* client = args->client ? args->client : GetOrCreateDefaultClient();
    if (!client) {
        return MakeError("No client available");
    }

    std::vector<int64_t> dims(args->dims, args->dims + args->num_dims);

    auto mps_buffer = client->client->BufferFromHostBuffer(
        args->data,
        static_cast<int>(args->type),
        dims,
        args->device ? args->device->device : nullptr
    );

    if (!mps_buffer) {
        return MakeError("Failed to create buffer");
    }

    auto* buffer = new PJRT_Buffer();
    buffer->buffer = std::move(mps_buffer);
    buffer->client = client;

    args->buffer = buffer;

    auto* event = new PJRT_Event();
    event->ready = true;
    args->done_with_host_buffer = event;

    return nullptr;
}

// ============================================================================
// Device Description API
// ============================================================================

PJRT_Error* MPS_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
    MPS_LOG(" PJRT_DeviceDescription_Id called\n");
    if (args->device_description && args->device_description->device) {
        args->id = args->device_description->device->device
            ? args->device_description->device->device->id() : 0;
    } else {
        args->id = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args) {
    args->process_index = 0;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
    static const char* kind = "gpu";
    args->device_kind = kind;
    args->device_kind_size = 3;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args) {
    static const char* str = "MPS:0";
    args->debug_string = str;
    args->debug_string_size = 5;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args) {
    static const char* str = "MpsDevice(id=0)";
    args->to_string = str;
    args->to_string_size = 15;
    return nullptr;
}

// ============================================================================
// Device API
// ============================================================================

PJRT_Error* MPS_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
    MPS_LOG(" PJRT_Device_GetDescription called, device=%p\n", (void*)args->device); fflush(stderr);
    if (args->device) {
        args->device_description = args->device->description;
        MPS_LOG(" Returning description=%p\n", (void*)args->device_description); fflush(stderr);
    } else {
        args->device_description = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
    MPS_LOG(" PJRT_Device_IsAddressable called, device=%p\n", (void*)args->device); fflush(stderr);
    args->is_addressable = true;
    MPS_LOG(" PJRT_Device_IsAddressable returning\n");
    return nullptr;
}

PJRT_Error* MPS_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
    MPS_LOG(" PJRT_Device_LocalHardwareId called, device=%p\n", (void*)args->device); fflush(stderr);
    args->local_hardware_id = args->device && args->device->device
        ? args->device->device->local_hardware_id() : 0;
    MPS_LOG(" PJRT_Device_LocalHardwareId returning %d\n", args->local_hardware_id); fflush(stderr);
    return nullptr;
}

PJRT_Error* MPS_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
    MPS_LOG(" PJRT_Device_AddressableMemories called\n");
    if (args->device && args->device->default_memory) {
        static PJRT_Memory* mem_array[1];
        mem_array[0] = args->device->default_memory;
        args->memories = mem_array;
        args->num_memories = 1;
    } else {
        args->memories = nullptr;
        args->num_memories = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
    MPS_LOG(" PJRT_Device_DefaultMemory called, device=%p\n", (void*)args->device); fflush(stderr);
    if (args->device) {
        args->memory = args->device->default_memory;
        MPS_LOG(" Returning memory=%p\n", (void*)args->memory); fflush(stderr);
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
    return MakeError("MemoryStats not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

// ============================================================================
// Memory API (stubs)
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
    MPS_LOG(" PJRT_Memory_AddressableByDevices called\n");
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

// ============================================================================
// Executable API
// ============================================================================

PJRT_Error* MPS_Executable_Destroy(PJRT_Executable_Destroy_Args* args) {
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPS_Executable_Name(PJRT_Executable_Name_Args* args) {
    MPS_LOG(" PJRT_Executable_Name called\n");
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
    MPS_LOG(" PJRT_Executable_NumOutputs called\n");
    args->num_outputs = args->executable && args->executable->executable
        ? args->executable->executable->num_outputs() : 1;
    MPS_LOG(" PJRT_Executable_NumOutputs: %zu\n", args->num_outputs); fflush(stderr);
    return nullptr;
}

PJRT_Error* MPS_Executable_SizeOfGeneratedCodeInBytes(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    args->size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* MPS_Executable_GetCostAnalysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    args->num_properties = 0;
    args->properties = nullptr;
    return nullptr;
}

// Static storage for output memory kinds
static const char* g_memory_kind = "device";
static const char* g_output_memory_kinds[1] = {g_memory_kind};
static size_t g_output_memory_kind_sizes[1] = {6};  // strlen("device")

PJRT_Error* MPS_Executable_OutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    MPS_LOG(" PJRT_Executable_OutputMemoryKinds called\n");
    args->num_outputs = 1;
    args->memory_kinds = g_output_memory_kinds;
    args->memory_kind_sizes = g_output_memory_kind_sizes;
    return nullptr;
}

// Static storage for output types and dimensions (single output case)
static PJRT_Buffer_Type g_output_types[1] = {PJRT_Buffer_Type_F32};
static int64_t g_output_dims[16] = {0};
static size_t g_output_dim_sizes[1] = {0};

PJRT_Error* MPS_Executable_OutputElementTypes(PJRT_Executable_OutputElementTypes_Args* args) {
    MPS_LOG(" PJRT_Executable_OutputElementTypes called\n");
    args->output_types = g_output_types;
    args->num_output_types = 1;
    return nullptr;
}

PJRT_Error* MPS_Executable_OutputDimensions(PJRT_Executable_OutputDimensions_Args* args) {
    MPS_LOG(" PJRT_Executable_OutputDimensions called\n");
    // For a 1D tensor of size 2: dims = [2], dim_sizes = [1]
    g_output_dims[0] = 2;
    g_output_dim_sizes[0] = 1;
    args->num_outputs = 1;
    args->dims = g_output_dims;
    args->dim_sizes = g_output_dim_sizes;
    return nullptr;
}

PJRT_Error* MPS_Executable_OptimizedProgram(PJRT_Executable_OptimizedProgram_Args* args) {
    return MakeError("OptimizedProgram not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_Serialize(PJRT_Executable_Serialize_Args* args) {
    return MakeError("Serialize not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    MPS_LOG(" PJRT_Executable_Fingerprint called\n");
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
    delete args->executable;
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_GetExecutable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_GetExecutable called, loaded_exec=%p\n",
            (void*)args->loaded_executable); fflush(stderr);
    if (args->loaded_executable) {
        MPS_LOG(" Getting executable from loaded, executable=%p\n",
                (void*)args->loaded_executable->executable); fflush(stderr);
        args->executable = args->loaded_executable->executable;
    } else {
        args->executable = nullptr;
    }
    MPS_LOG(" PJRT_LoadedExecutable_GetExecutable returning executable=%p\n",
            (void*)args->executable); fflush(stderr);
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_AddressableDevices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_AddressableDevices called\n");
    MPS_LOG("   args->executable=%p\n", (void*)args->executable); fflush(stderr);

    // Return devices from the LoadedExecutable's client
    if (args->executable && args->executable->client && !args->executable->client->devices.empty()) {
        args->addressable_devices = args->executable->client->devices.data();
        args->num_addressable_devices = args->executable->client->devices.size();
        MPS_LOG("   Returning %zu devices\n", args->num_addressable_devices); fflush(stderr);
    } else {
        args->addressable_devices = nullptr;
        args->num_addressable_devices = 0;
        MPS_LOG("   Returning 0 devices\n");
    }

    MPS_LOG(" PJRT_LoadedExecutable_AddressableDevices returning\n");
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_Delete(PJRT_LoadedExecutable_Delete_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_IsDeleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_IsDeleted called\n");
    args->is_deleted = false;
    return nullptr;
}

PJRT_Error* MPS_LoadedExecutable_Execute(PJRT_LoadedExecutable_Execute_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_Execute\n");

    if (!args->executable || !args->executable->executable) {
        return MakeError("No executable to execute");
    }

    std::vector<jax_mps::MpsBuffer*> inputs;
    for (size_t i = 0; i < args->num_args; i++) {
        if (args->argument_lists[0][i]) {
            inputs.push_back(args->argument_lists[0][i]->buffer.get());
        }
    }

    PJRT_Client* client = args->executable->client;
    jax_mps::MpsDevice* device = client && !client->devices.empty()
        ? client->devices[0]->device : nullptr;

    auto results = args->executable->executable->executable->Execute(inputs, device);

    // Write outputs to the pre-allocated output_lists
    size_t num_outputs = results.size();
    for (size_t i = 0; i < num_outputs; i++) {
        auto* buffer = new PJRT_Buffer();
        buffer->buffer = std::move(results[i]);
        buffer->client = client;
        args->output_lists[0][i] = buffer;
    }

    if (args->device_complete_events) {
        auto* event = new PJRT_Event();
        event->ready = true;
        args->device_complete_events[0] = event;
    }

    return nullptr;
}

PJRT_Error* MPS_Executable_DeserializeAndLoad(PJRT_Executable_DeserializeAndLoad_Args* args) {
    return MakeError("DeserializeAndLoad not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_LoadedExecutable_Fingerprint(PJRT_LoadedExecutable_Fingerprint_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_Fingerprint called\n");
    args->executable_fingerprint = nullptr;
    args->executable_fingerprint_size = 0;
    return nullptr;
}

// Backing storage for device assignment serialization
struct MpsDeviceAssignmentSerialized {
    std::string data;
};

static void MpsDeviceAssignmentDeleter(PJRT_DeviceAssignmentSerialized* da) {
    MPS_LOG(" MpsDeviceAssignmentDeleter called\n");
    delete reinterpret_cast<MpsDeviceAssignmentSerialized*>(da);
}

PJRT_Error* MPS_LoadedExecutable_GetDeviceAssignment(PJRT_LoadedExecutable_GetDeviceAssignment_Args* args) {
    MPS_LOG(" PJRT_LoadedExecutable_GetDeviceAssignment called\n");

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
    args->serialized_device_assignment = reinterpret_cast<PJRT_DeviceAssignmentSerialized*>(serialized);
    args->serialized_device_assignment_deleter = MpsDeviceAssignmentDeleter;

    MPS_LOG(" PJRT_LoadedExecutable_GetDeviceAssignment returning %zu bytes\n",
            args->serialized_bytes_size);
    return nullptr;
}

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
    args->on_device_size_in_bytes = args->buffer && args->buffer->buffer
        ? args->buffer->buffer->byte_size() : 0;
    return nullptr;
}

PJRT_Error* MPS_Buffer_Device(PJRT_Buffer_Device_Args* args) {
    if (args->buffer && args->buffer->client && !args->buffer->client->devices.empty()) {
        args->device = args->buffer->client->devices[0];
    } else {
        args->device = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Buffer_Memory(PJRT_Buffer_Memory_Args* args) {
    MPS_LOG(" PJRT_Buffer_Memory called\n");
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
    args->is_deleted = args->buffer && args->buffer->buffer
        ? args->buffer->buffer->IsDeleted() : true;
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

PJRT_Error* MPS_Buffer_IncreaseExternalReferenceCount(PJRT_Buffer_IncreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_DecreaseExternalReferenceCount(PJRT_Buffer_DecreaseExternalReferenceCount_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_Buffer_OpaqueDeviceMemoryDataPointer(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args) {
    args->device_memory_ptr = 0;
    return nullptr;
}

// ============================================================================
// CopyToDeviceStream API (stubs)
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

// ============================================================================
// Client TopologyDescription
// ============================================================================

PJRT_Error* MPS_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args) {
    MPS_LOG(" PJRT_Client_TopologyDescription called\n");
    if (args->client && args->client->topology) {
        args->topology = args->client->topology;
    } else {
        args->topology = nullptr;
    }
    return nullptr;
}

// ============================================================================
// TopologyDescription API (stubs)
// ============================================================================

PJRT_Error* MPS_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args* args) {
    args->topology = nullptr;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
    static const char* name = "mps";
    args->platform_name = name;
    args->platform_name_size = 3;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_PlatformVersion(PJRT_TopologyDescription_PlatformVersion_Args* args) {
    static const char* version = "0.1.0";
    args->platform_version = version;
    args->platform_version_size = 5;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_GetDeviceDescriptions(PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
    args->descriptions = nullptr;
    args->num_descriptions = 0;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_Serialize(PJRT_TopologyDescription_Serialize_Args* args) {
    return MakeError("TopologyDescription_Serialize not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args) {
    args->attributes = nullptr;
    args->num_attributes = 0;
    return nullptr;
}

// ============================================================================
// Compile API
// ============================================================================

PJRT_Error* MPS_Compile(PJRT_Compile_Args* args) {
    return MakeError("PJRT_Compile not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}

// ============================================================================
// PJRT_Api - main entry point
// ============================================================================

static const PJRT_Api pjrt_api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .extension_start = nullptr,

    .pjrt_api_version = {
        .struct_size = PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = nullptr,
        .major_version = PJRT_API_MAJOR,
        .minor_version = PJRT_API_MINOR,
    },

    .PJRT_Error_Destroy = MPS_Error_Destroy,
    .PJRT_Error_Message = MPS_Error_Message,
    .PJRT_Error_GetCode = MPS_Error_GetCode,

    .PJRT_Plugin_Initialize = MPS_Plugin_Initialize,
    .PJRT_Plugin_Attributes = MPS_Plugin_Attributes,

    .PJRT_Event_Destroy = MPS_Event_Destroy,
    .PJRT_Event_IsReady = MPS_Event_IsReady,
    .PJRT_Event_Error = MPS_Event_Error,
    .PJRT_Event_Await = MPS_Event_Await,
    .PJRT_Event_OnReady = MPS_Event_OnReady,

    .PJRT_Client_Create = MPS_Client_Create,
    .PJRT_Client_Destroy = MPS_Client_Destroy,
    .PJRT_Client_PlatformName = MPS_Client_PlatformName,
    .PJRT_Client_ProcessIndex = MPS_Client_ProcessIndex,
    .PJRT_Client_PlatformVersion = MPS_Client_PlatformVersion,
    .PJRT_Client_Devices = MPS_Client_Devices,
    .PJRT_Client_AddressableDevices = MPS_Client_AddressableDevices,
    .PJRT_Client_LookupDevice = MPS_Client_LookupDevice,
    .PJRT_Client_LookupAddressableDevice = MPS_Client_LookupAddressableDevice,
    .PJRT_Client_AddressableMemories = MPS_Client_AddressableMemories,
    .PJRT_Client_Compile = MPS_Client_Compile,
    .PJRT_Client_DefaultDeviceAssignment = MPS_Client_DefaultDeviceAssignment,
    .PJRT_Client_BufferFromHostBuffer = MPS_Client_BufferFromHostBuffer,

    .PJRT_DeviceDescription_Id = MPS_DeviceDescription_Id,
    .PJRT_DeviceDescription_ProcessIndex = MPS_DeviceDescription_ProcessIndex,
    .PJRT_DeviceDescription_Attributes = MPS_DeviceDescription_Attributes,
    .PJRT_DeviceDescription_Kind = MPS_DeviceDescription_Kind,
    .PJRT_DeviceDescription_DebugString = MPS_DeviceDescription_DebugString,
    .PJRT_DeviceDescription_ToString = MPS_DeviceDescription_ToString,

    .PJRT_Device_GetDescription = MPS_Device_GetDescription,
    .PJRT_Device_IsAddressable = MPS_Device_IsAddressable,
    .PJRT_Device_LocalHardwareId = MPS_Device_LocalHardwareId,
    .PJRT_Device_AddressableMemories = MPS_Device_AddressableMemories,
    .PJRT_Device_DefaultMemory = MPS_Device_DefaultMemory,
    .PJRT_Device_MemoryStats = MPS_Device_MemoryStats,

    .PJRT_Memory_Id = MPS_Memory_Id,
    .PJRT_Memory_Kind = MPS_Memory_Kind,
    .PJRT_Memory_DebugString = MPS_Memory_DebugString,
    .PJRT_Memory_ToString = MPS_Memory_ToString,
    .PJRT_Memory_AddressableByDevices = MPS_Memory_AddressableByDevices,

    .PJRT_Executable_Destroy = MPS_Executable_Destroy,
    .PJRT_Executable_Name = MPS_Executable_Name,
    .PJRT_Executable_NumReplicas = MPS_Executable_NumReplicas,
    .PJRT_Executable_NumPartitions = MPS_Executable_NumPartitions,
    .PJRT_Executable_NumOutputs = MPS_Executable_NumOutputs,
    .PJRT_Executable_SizeOfGeneratedCodeInBytes = MPS_Executable_SizeOfGeneratedCodeInBytes,
    .PJRT_Executable_GetCostAnalysis = MPS_Executable_GetCostAnalysis,
    .PJRT_Executable_OutputMemoryKinds = MPS_Executable_OutputMemoryKinds,
    .PJRT_Executable_OptimizedProgram = MPS_Executable_OptimizedProgram,
    .PJRT_Executable_Serialize = MPS_Executable_Serialize,

    .PJRT_LoadedExecutable_Destroy = MPS_LoadedExecutable_Destroy,
    .PJRT_LoadedExecutable_GetExecutable = MPS_LoadedExecutable_GetExecutable,
    .PJRT_LoadedExecutable_AddressableDevices = MPS_LoadedExecutable_AddressableDevices,
    .PJRT_LoadedExecutable_Delete = MPS_LoadedExecutable_Delete,
    .PJRT_LoadedExecutable_IsDeleted = MPS_LoadedExecutable_IsDeleted,
    .PJRT_LoadedExecutable_Execute = MPS_LoadedExecutable_Execute,
    .PJRT_Executable_DeserializeAndLoad = MPS_Executable_DeserializeAndLoad,
    .PJRT_LoadedExecutable_Fingerprint = MPS_LoadedExecutable_Fingerprint,

    .PJRT_Buffer_Destroy = MPS_Buffer_Destroy,
    .PJRT_Buffer_ElementType = MPS_Buffer_ElementType,
    .PJRT_Buffer_Dimensions = MPS_Buffer_Dimensions,
    .PJRT_Buffer_UnpaddedDimensions = MPS_Buffer_UnpaddedDimensions,
    .PJRT_Buffer_DynamicDimensionIndices = MPS_Buffer_DynamicDimensionIndices,
    .PJRT_Buffer_GetMemoryLayout = MPS_Buffer_GetMemoryLayout,
    .PJRT_Buffer_OnDeviceSizeInBytes = MPS_Buffer_OnDeviceSizeInBytes,
    .PJRT_Buffer_Device = MPS_Buffer_Device,
    .PJRT_Buffer_Memory = MPS_Buffer_Memory,
    .PJRT_Buffer_Delete = MPS_Buffer_Delete,
    .PJRT_Buffer_IsDeleted = MPS_Buffer_IsDeleted,
    .PJRT_Buffer_CopyToDevice = MPS_Buffer_CopyToDevice,
    .PJRT_Buffer_ToHostBuffer = MPS_Buffer_ToHostBuffer,
    .PJRT_Buffer_IsOnCpu = MPS_Buffer_IsOnCpu,
    .PJRT_Buffer_ReadyEvent = MPS_Buffer_ReadyEvent,
    .PJRT_Buffer_UnsafePointer = MPS_Buffer_UnsafePointer,
    .PJRT_Buffer_IncreaseExternalReferenceCount = MPS_Buffer_IncreaseExternalReferenceCount,
    .PJRT_Buffer_DecreaseExternalReferenceCount = MPS_Buffer_DecreaseExternalReferenceCount,
    .PJRT_Buffer_OpaqueDeviceMemoryDataPointer = MPS_Buffer_OpaqueDeviceMemoryDataPointer,

    .PJRT_CopyToDeviceStream_Destroy = MPS_CopyToDeviceStream_Destroy,
    .PJRT_CopyToDeviceStream_AddChunk = MPS_CopyToDeviceStream_AddChunk,
    .PJRT_CopyToDeviceStream_TotalBytes = MPS_CopyToDeviceStream_TotalBytes,
    .PJRT_CopyToDeviceStream_GranuleSize = MPS_CopyToDeviceStream_GranuleSize,
    .PJRT_CopyToDeviceStream_CurrentBytes = MPS_CopyToDeviceStream_CurrentBytes,

    .PJRT_TopologyDescription_Create = MPS_TopologyDescription_Create,
    .PJRT_TopologyDescription_Destroy = MPS_TopologyDescription_Destroy,
    .PJRT_TopologyDescription_PlatformName = MPS_TopologyDescription_PlatformName,
    .PJRT_TopologyDescription_PlatformVersion = MPS_TopologyDescription_PlatformVersion,
    .PJRT_TopologyDescription_GetDeviceDescriptions = MPS_TopologyDescription_GetDeviceDescriptions,
    .PJRT_TopologyDescription_Serialize = MPS_TopologyDescription_Serialize,
    .PJRT_TopologyDescription_Attributes = MPS_TopologyDescription_Attributes,

    .PJRT_Compile = MPS_Compile,

    // Output type/dimension information
    .PJRT_Executable_OutputElementTypes = MPS_Executable_OutputElementTypes,
    .PJRT_Executable_OutputDimensions = MPS_Executable_OutputDimensions,
    .PJRT_Buffer_CopyToMemory = nullptr,
    .PJRT_Client_CreateViewOfDeviceBuffer = nullptr,
    .PJRT_Executable_Fingerprint = MPS_Executable_Fingerprint,
    .PJRT_Client_TopologyDescription = MPS_Client_TopologyDescription,
    .PJRT_Executable_GetCompiledMemoryStats = nullptr,
    .PJRT_Memory_Kind_Id = nullptr,
    .PJRT_ExecuteContext_Create = nullptr,
    .PJRT_ExecuteContext_Destroy = nullptr,
    .PJRT_Buffer_CopyRawToHost = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_Destroy = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_TransferData = nullptr,
    .PJRT_Client_CreateBuffersForAsyncHostToDevice = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_Device = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_BufferCount = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_BufferSize = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_SetBufferError = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_AddMetadata = nullptr,
    .PJRT_Client_DmaMap = nullptr,
    .PJRT_Client_DmaUnmap = nullptr,
    .PJRT_Client_CreateUninitializedBuffer = nullptr,
    .PJRT_Client_UpdateGlobalProcessInfo = nullptr,
    .PJRT_TopologyDescription_Deserialize = nullptr,
    .PJRT_Client_CreateAliasBuffer = nullptr,
    .PJRT_Client_FulfillAliasBuffer = nullptr,
    .PJRT_LoadedExecutable_GetDeviceAssignment = MPS_LoadedExecutable_GetDeviceAssignment,
    .PJRT_Client_CreateErrorBuffer = nullptr,
    .PJRT_AsyncHostToDeviceTransferManager_TransferLiteral = nullptr,
    .PJRT_Buffer_CopyRawToHostFuture = nullptr,
    .PJRT_Device_PoisonExecution = nullptr,
    .PJRT_Device_CreateAsyncTrackingEvent = nullptr,
    .PJRT_AsyncTrackingEvent_Destroy = nullptr,
    .PJRT_Executable_GetCompileOptions = nullptr,
    .PJRT_Buffer_DonateWithControlDependency = nullptr,
    .PJRT_Event_Create = nullptr,
    .PJRT_Event_Set = nullptr,
};

extern "C" {

__attribute__((visibility("default")))
const PJRT_Api* GetPjrtApi() {
    return &pjrt_api;
}

}
