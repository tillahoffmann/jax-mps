// Minimal PJRT profiler extension — returns errors for all operations.
#pragma once

#include <xla/backends/profiler/plugin/profiler_c_api.h>
#include <xla/pjrt/c/pjrt_c_api_profiler_extension.h>

// Returns the profiler extension base pointer to register in the PJRT API.
PJRT_Extension_Base* GetProfilerExtension();
