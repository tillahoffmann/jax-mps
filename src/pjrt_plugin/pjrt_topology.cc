// PJRT TopologyDescription API implementation for Metal backend

#include "pjrt_plugin/logging.h"
#include "pjrt_plugin/pjrt_types.h"

// ============================================================================
// Client TopologyDescription
// ============================================================================

PJRT_Error* MPS_Client_TopologyDescription(PJRT_Client_TopologyDescription_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_TopologyDescription called\n");
    if (args->client && args->client->topology) {
        args->topology = args->client->topology;
    } else {
        args->topology = nullptr;
    }
    return nullptr;
}

// ============================================================================
// TopologyDescription API
// ============================================================================

PJRT_Error* MPS_TopologyDescription_Create(PJRT_TopologyDescription_Create_Args* args) {
    args->topology = nullptr;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_Destroy(PJRT_TopologyDescription_Destroy_Args* args) {
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_PlatformName(PJRT_TopologyDescription_PlatformName_Args* args) {
    args->platform_name = kPlatformName;
    args->platform_name_size = strlen(kPlatformName);
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args) {
    args->platform_version = kPlatformVersion;
    args->platform_version_size = strlen(kPlatformVersion);
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args) {
    args->descriptions = nullptr;
    args->num_descriptions = 0;
    return nullptr;
}

PJRT_Error* MPS_TopologyDescription_Serialize(PJRT_TopologyDescription_Serialize_Args* args) {
    return MakeError("TopologyDescription_Serialize not implemented",
                     PJRT_Error_Code_UNIMPLEMENTED);
}

PJRT_Error* MPS_TopologyDescription_Attributes(PJRT_TopologyDescription_Attributes_Args* args) {
    args->attributes = nullptr;
    args->num_attributes = 0;
    return nullptr;
}
