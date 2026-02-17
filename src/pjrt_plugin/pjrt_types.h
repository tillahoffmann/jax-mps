// PJRT opaque wrapper types for Metal backend
#pragma once

#include <xla/pjrt/c/pjrt_c_api.h>

#include <memory>
#include <string>
#include <vector>

#include "pjrt_plugin/mps_buffer.h"
#include "pjrt_plugin/mps_client.h"
#include "pjrt_plugin/mps_device.h"
#include "pjrt_plugin/mps_executable.h"

// ============================================================================
// Forward declarations
// ============================================================================

struct PJRT_TopologyDescription;
struct PJRT_DeviceDescription;
struct PJRT_Memory;

// ============================================================================
// Opaque wrapper types
// ============================================================================

struct PJRT_Client {
    std::unique_ptr<jax_mps::MpsClient> client;
    std::vector<PJRT_Device*> devices;
    std::vector<PJRT_Memory*> memories;
    PJRT_TopologyDescription* topology;
};

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

    // Ownership flag: when true, this executable is owned by a PJRT_LoadedExecutable
    // and should not be deleted directly by PJRT_Executable_Destroy
    bool owned_by_loaded = false;

    // Dynamic storage for output metadata (lazily initialized)
    mutable std::vector<const char*> output_memory_kinds;
    mutable std::vector<size_t> output_memory_kind_sizes;
    mutable std::vector<PJRT_Buffer_Type> output_types;
    mutable std::vector<int64_t> output_dims;
    mutable std::vector<size_t> output_dim_sizes;
    mutable bool output_metadata_initialized = false;

    void initOutputMetadata() const {
        if (output_metadata_initialized)
            return;
        size_t num_outputs = executable ? executable->num_outputs() : 1;
        output_memory_kinds.resize(num_outputs, "device");
        output_memory_kind_sizes.resize(num_outputs, 6);  // strlen("device")
        output_types.resize(num_outputs, PJRT_Buffer_Type_F32);
        output_dims.resize(num_outputs * 8, 0);  // up to 8 dims per output
        output_dim_sizes.resize(num_outputs, 0);
        output_metadata_initialized = true;
    }
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
// Error type
// ============================================================================

struct PJRT_Error {
    std::string message;
    PJRT_Error_Code code;
};

// ============================================================================
// Helper functions
// ============================================================================

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL);

// Global client management
PJRT_Client* GetOrCreateDefaultClient();
PJRT_Client* GetClient(PJRT_Client* client);

// Platform constants
extern const char* const kPlatformName;
extern const char* const kPlatformVersion;
