#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps_forward", &furthest_point_sampling_wrapper, "FPS forward (CUDA)");
}