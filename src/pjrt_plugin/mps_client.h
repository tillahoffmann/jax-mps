#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward declarations for Metal types (defined in .mm files)
#ifdef __OBJC__
@class MTLDevice;
@class MPSGraph;
#else
typedef void* MTLDevice;
typedef void* MPSGraph;
#endif

namespace mps {
struct StableHLOModule;
}

namespace jax_mps {

class MpsDevice;
class MpsBuffer;
class MpsExecutable;

// Represents a Metal GPU client for PJRT
class MpsClient {
public:
    static std::unique_ptr<MpsClient> Create();
    ~MpsClient();

    // Client info
    const std::string& platform_name() const {
        return platform_name_;
    }
    const std::string& platform_version() const {
        return platform_version_;
    }
    int process_index() const {
        return 0;
    }  // Single process

    // Device management
    int device_count() const;
    int addressable_device_count() const;
    MpsDevice* device(int index);
    MpsDevice* addressable_device(int index);
    MpsDevice* LookupDevice(int device_id);

    // Buffer operations
    std::unique_ptr<MpsBuffer> BufferFromHostBuffer(const void* data,
                                                    int dtype,  // PJRT dtype enum
                                                    const std::vector<int64_t>& dims,
                                                    MpsDevice* device);

    // Compilation
    std::unique_ptr<MpsExecutable> CompileStableHLO(const mps::StableHLOModule& module,
                                                    MpsDevice* device);

    // Internal: get Metal device
    void* metal_device() const {
        return metal_device_;
    }

private:
    MpsClient();
    bool Initialize();

    std::string platform_name_;
    std::string platform_version_;
    void* metal_device_;  // id<MTLDevice>
    std::vector<std::unique_ptr<MpsDevice>> devices_;
};

}  // namespace jax_mps
