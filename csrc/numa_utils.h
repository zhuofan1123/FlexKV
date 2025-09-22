#pragma once

#include <numa.h>
#include <torch/torch.h>
#include <vector>

namespace flexkv {

bool numa_is_available();

int get_numa_node_count();

int verify_memory_node_tensor(const torch::Tensor &tensor);

void numa_free_memory(void *ptr, size_t size);

torch::Tensor create_tensor_with_numa_bind(int node,
                                           const std::vector<int64_t> &shape,
                                           torch::ScalarType dtype);

} // namespace flexkv
