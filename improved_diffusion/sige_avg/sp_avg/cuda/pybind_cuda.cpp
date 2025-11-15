#include "gather.cpp"
#include "scatter.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Sparse Process Engine (SPE)";
    m.def("gather", &gather_cuda, "Gather (CUDA)");
    m.def("scatter", &scatter_cuda, "Scatter (CUDA)");
    m.def("scatter_avg", &scatter_avg_cuda, "Scatter with average (CUDA)");
}