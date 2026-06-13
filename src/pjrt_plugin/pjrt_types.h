// PJRT opaque wrapper types for Metal backend
#pragma once

#include <mlx/event.h>
#include <xla/pjrt/c/pjrt_c_api.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "pjrt_plugin/mlx_buffer.h"
#include "pjrt_plugin/mlx_client.h"
#include "pjrt_plugin/mlx_device.h"
#include "pjrt_plugin/mlx_executable.h"

// ============================================================================
// Forward declarations
// ============================================================================

struct PJRT_TopologyDescription;
struct PJRT_DeviceDescription;
struct PJRT_Memory;
struct PJRT_Error;

// ============================================================================
// Opaque wrapper types
// ============================================================================

struct PJRT_Client {
    std::unique_ptr<jax_mps::MlxClient> client;
    std::vector<PJRT_Device*> devices;
    std::vector<PJRT_Memory*> memories;
    PJRT_TopologyDescription* topology;
};

struct PJRT_Device {
    jax_mps::MlxDevice* device;
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
    std::unique_ptr<jax_mps::MlxBuffer> buffer;
    PJRT_Client* client;
};

struct PJRT_Executable {
    std::unique_ptr<jax_mps::MlxExecutable> executable;
    PJRT_Client* client;

    // Ownership flag: when true, this executable is owned by a PJRT_LoadedExecutable
    // and should not be deleted directly by PJRT_Executable_Destroy
    bool owned_by_loaded = false;

    // Dynamic storage for output metadata (lazily initialized, thread-safe)
    mutable std::vector<const char*> output_memory_kinds;
    mutable std::vector<size_t> output_memory_kind_sizes;
    mutable std::vector<PJRT_Buffer_Type> output_types;
    mutable std::vector<int64_t> output_dims;
    mutable std::vector<size_t> output_dim_sizes;
    mutable std::once_flag output_metadata_flag;

    void initOutputMetadata() const {
        std::call_once(output_metadata_flag, [this] {
            size_t num_outputs = executable ? executable->num_outputs() : 1;
            output_memory_kinds.resize(num_outputs, "device");
            output_memory_kind_sizes.resize(num_outputs, 6);  // strlen("device")
            output_types.resize(num_outputs, PJRT_Buffer_Type_F32);
            output_dims.resize(num_outputs * 8, 0);  // up to 8 dims per output
            output_dim_sizes.resize(num_outputs, 0);
        });
    }
};

struct PJRT_LoadedExecutable {
    PJRT_Executable* executable;
    PJRT_Client* client;
    std::vector<PJRT_Device*> addressable_devices;
};

// A PJRT completion event backed by MLX stream events.
//
// Under async dispatch (JAX_MPS_ASYNC_DISPATCH), Execute() returns before the
// GPU finishes, so PJRT events must report real completion. We snapshot a copy
// of each tracked array's MLX `Event` (the underlying MTLSharedEvent) at
// construction time. All readiness queries operate ONLY on these Event copies —
// never on the mlx::core::array / array_desc — because the array's `status`
// field is a plain enum with no synchronization and may be mutated by the
// scheduler or by a subsequent Execute() on the main thread. The MLX Event, by
// contrast, wraps an MTLSharedEvent that is safe to wait/poll from any thread.
//
// In synchronous mode (flag off) the tracked arrays are already evaluated, so
// `events_` is empty (or the events are already signaled) and the object
// behaves as immediately-ready — preserving the previous behavior exactly.
struct PJRT_Event {
    // Trivially-ready event (no work to wait on).
    PJRT_Event() = default;

    // Track completion of the given arrays. Copies their valid MLX events.
    explicit PJRT_Event(std::vector<mlx::core::array> arrays);

    // Non-blocking: true once all tracked work has completed.
    bool IsReady();

    // Blocks the calling thread until all tracked work has completed.
    void Await();

    // Registers `callback` to fire once ready. If already ready it is invoked
    // inline; otherwise a shared background thread waits and then invokes it.
    void OnReady(PJRT_Event_OnReadyCallback callback, void* user_arg);

    std::mutex mu_;
    std::vector<mlx::core::Event> events_;
    bool ready_ = true;
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
