#pragma once

#include <string>

namespace jax_mps {

class MpsClient;

// Represents a single Metal GPU device
class MpsDevice {
public:
    MpsDevice(MpsClient* client, int id, const std::string& name);
    ~MpsDevice();

    // Device identification
    int id() const {
        return id_;
    }
    int local_hardware_id() const {
        return id_;
    }
    const std::string& device_kind() const {
        return device_kind_;
    }
    const std::string& debug_string() const {
        return debug_string_;
    }
    const std::string& to_string() const {
        return debug_string_;
    }

    // Owning client
    MpsClient* client() const {
        return client_;
    }

    // Device is always addressable in single-process mode
    bool IsAddressable() const {
        return true;
    }

private:
    MpsClient* client_;
    int id_;
    std::string device_kind_;
    std::string debug_string_;
};

}  // namespace jax_mps
