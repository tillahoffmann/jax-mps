#pragma once

// ============================================================================
// Logging macros for MPS PJRT plugin
//
// Log levels (set MPS_LOG_LEVEL before including this header):
//   0 = ERROR only
//   1 = ERROR + WARN (default)
//   2 = ERROR + WARN + INFO
//   3 = ERROR + WARN + INFO + DEBUG
// ============================================================================

#include <cstdio>

#ifndef MPS_LOG_LEVEL
#define MPS_LOG_LEVEL 1
#endif

// Level 0+: Errors (operation failures, invalid states)
#if MPS_LOG_LEVEL >= 0
#define MPS_LOG_ERROR(...)                           \
    do {                                             \
        fprintf(stderr, "[MPS ERROR] " __VA_ARGS__); \
    } while (0)
#else
#define MPS_LOG_ERROR(...) \
    do {                   \
    } while (0)
#endif

// Level 1+: Warnings (unexpected but recoverable)
#if MPS_LOG_LEVEL >= 1
#define MPS_LOG_WARN(...)                           \
    do {                                            \
        fprintf(stderr, "[MPS WARN] " __VA_ARGS__); \
    } while (0)
#else
#define MPS_LOG_WARN(...) \
    do {                  \
    } while (0)
#endif

// Level 2+: Info (operation flow, milestones)
#if MPS_LOG_LEVEL >= 2
#define MPS_LOG_INFO(...)                           \
    do {                                            \
        fprintf(stderr, "[MPS INFO] " __VA_ARGS__); \
    } while (0)
#else
#define MPS_LOG_INFO(...) \
    do {                  \
    } while (0)
#endif

// Level 3+: Debug (detailed data, addresses, shapes)
#if MPS_LOG_LEVEL >= 3
#define MPS_LOG_DEBUG(...)                           \
    do {                                             \
        fprintf(stderr, "[MPS DEBUG] " __VA_ARGS__); \
    } while (0)
#else
#define MPS_LOG_DEBUG(...) \
    do {                   \
    } while (0)
#endif
