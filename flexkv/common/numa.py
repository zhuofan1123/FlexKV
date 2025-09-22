import time
from typing import NewType, Optional

import numpy as np
import torch

from flexkv import c_ext

def numa_is_available() -> bool:
    return c_ext.numa_is_available()

def get_numa_node_count() -> int:
    return c_ext.get_numa_node_count()

def verify_memory_node_tensor(tensor: torch.Tensor) -> int:
    return c_ext.verify_memory_node_tensor(tensor)

def create_tensor_with_numa_bind(node: int, shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    return c_ext.create_tensor_with_numa_bind(node, shape, dtype)

if __name__ == "__main__":
    try:
        print(numa_is_available())
        print(get_numa_node_count())
        a = create_tensor_with_numa_bind(0, [1, 2, 3], torch.float32)
        print(verify_memory_node_tensor(a))
    except Exception as e:
        print(e)
