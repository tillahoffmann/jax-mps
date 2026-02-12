#pragma once

// Control flow ops (stablehlo.while, stablehlo.case) are registered as normal
// GRAPH ops via OpRegistry. No external declarations needed - handlers are
// file-local in control_flow_ops.mm.
