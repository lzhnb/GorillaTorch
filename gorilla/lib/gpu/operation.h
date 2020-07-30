#include <torch/extension.h>
#include <vector>

at::Tensor furthest_point_sampling(
    int num_sample,
    const at::Tensor& input
);

