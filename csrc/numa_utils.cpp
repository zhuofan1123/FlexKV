#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numa.h>
#include <numaif.h>
#include <torch/torch.h>
#include <vector>

#include "numa_utils.h"

namespace flexkv {

/**
 * Check if NUMA is available
 * @return true if NUMA is available, false otherwise
 */
bool numa_is_available() { return numa_available() >= 0; }

/**
 * Get the number of NUMA nodes
 * @return The number of NUMA nodes, if NUMA is not available return 0
 */
int get_numa_node_count() { return numa_max_node() + 1; }

/**
 * Allocate memory on a specific numa node with strict binding
 * @param node Target NUMA node number
 * @param size Memory size to allocate (in bytes)
 * @return Memory pointer on success, nullptr on failure
 */
static void *numa_alloc_strict_bind(int node, size_t size) {
  if (!numa_is_available()) {
    return nullptr;
  }

  if (node < 0 || node > numa_max_node()) {
    return nullptr;
  }

  if (!numa_bitmask_isbitset(numa_get_mems_allowed(), node)) {
    return nullptr;
  }

  int old_policy;
  struct bitmask *old_nodemask = numa_allocate_nodemask();
  get_mempolicy(&old_policy, old_nodemask->maskp, old_nodemask->size + 1,
                nullptr, 0);

  struct bitmask *nodemask = numa_allocate_nodemask();
  numa_bitmask_clearall(nodemask);
  numa_bitmask_setbit(nodemask, node);

  int result = set_mempolicy(MPOL_BIND, nodemask->maskp, nodemask->size + 1);

  void *ptr = nullptr;
  if (result == 0) {
    ptr = numa_alloc(size);

    if (ptr) {
      memset(ptr, 0, size);

      int allocated_node = -1;
      if (get_mempolicy(&allocated_node, nullptr, 0, ptr,
                        MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        if (allocated_node != node) {
          numa_free(ptr, size);
          ptr = nullptr;
        }
      }
    }
  }

  set_mempolicy(old_policy, old_nodemask->maskp, old_nodemask->size + 1);

  numa_free_nodemask(nodemask);
  numa_free_nodemask(old_nodemask);

  return ptr;
}

/**
 * Verify if memory is allocated on the specified numa node
 * @param ptr Memory pointer
 * @return Node number where memory is allocated, -1 on failure
 */
int verify_memory_node(void *ptr) {
  int allocated_node = -1;
  if (get_mempolicy(&allocated_node, nullptr, 0, ptr,
                    MPOL_F_NODE | MPOL_F_ADDR) == 0) {
    return allocated_node;
  }
  return -1;
}

/**
 * Given a PyTorch tensor, return the NUMA node number it is allocated on
 * @param tensor Input PyTorch tensor
 * @return NUMA node number where tensor is allocated, -1 on failure
 */
int verify_memory_node_tensor(const torch::Tensor &tensor) {
  if (!tensor.device().is_cpu()) {
    throw std::runtime_error("verify_memory_node only supports CPU tensor");
  }
  void *ptr = tensor.data_ptr();
  return verify_memory_node(ptr);
}

/**
 * Free NUMA memory
 * @param ptr Memory pointer to free
 * @param size Memory size (in bytes)
 */
void numa_free_memory(void *ptr, size_t size) {
  if (ptr) {
    numa_free(ptr, size);
  }
}

/**
 * Create a PyTorch tensor on specified NUMA node
 * @param node Target NUMA node number
 * @param shape Vector of tensor dimensions
 * @param dtype Data type (default: float32)
 * @return PyTorch tensor with automatic memory management on specified NUMA
 * node
 */
torch::Tensor create_tensor_with_numa_bind(int node,
                                           const std::vector<int64_t> &shape,
                                           torch::ScalarType dtype) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("Invalid dimension size in shape");
    }
    total_elements *= dim;
  }

  size_t element_size = torch::elementSize(dtype);
  size_t actual_size = total_elements * element_size;

  size_t alloc_size = actual_size;

  void *ptr = numa_alloc_strict_bind(node, alloc_size);
  if (!ptr) {
    throw std::runtime_error("Failed to allocate memory on NUMA node " +
                             std::to_string(node));
  }

  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

  auto deleter = [alloc_size, node](void *ptr) {
    if (ptr) {
      numa_free(ptr, alloc_size);
    }
  };

  torch::Tensor tensor = torch::from_blob(ptr, shape, deleter, options);

  return tensor;
}

} // namespace flexkv
