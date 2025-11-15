#include "gather.h"
#include "scatter.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Process Engine (SPE)";
    m.def("gather", &gather_cpu, "Gather (CPU)");
    m.def("scatter", &scatter_cpu, "Scatter (CPU)");
    m.def("scatter_avg", &scatter_avg_cpu, "Scatter with average (CPU)");
}
