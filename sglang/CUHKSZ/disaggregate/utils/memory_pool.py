import torch
import threading
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class MoEMemoryPool:    
    def __init__(self, 
                 total_size_mb: int = 1536,  # default 1.5GB # TODO 从config读取
                 device: str = "cuda"):
        self.device = device
        self.total_size_mb = total_size_mb
        self.allocated_bytes = 0
        self.lock = threading.Lock()
        
        # for storing free tensors of different shapes
        # key is (shape, dtype), value is list of available tensors
        self.free_tensors: Dict[Tuple, List[torch.Tensor]] = {}
        
        # preallocate some common size tensors
        self._preallocate_common_tensors()
        
        logger.info(f"MoEMemoryPool initialized with {total_size_mb}MB on {device}")
    
    def _preallocate_common_tensors(self):        
        # preallocate some common size hidden_state tensors
        hidden_dims = [2048, 5120, 7168] # TODO
        router_logits_num = [60, 160, 256] # TODO 这两个值应该根据config具体决定
        # router_logits
        batch_sizes = [1, 2, 4] # ? 如果不聚合,则可能浪费掉了. 或者根据hidden_dim直接划分成1*hidden_dim的,需要就从中拿
        
        # preallocate hidden_state tensors
        # for hidden_dim in hidden_dims:
        #     for batch_size in batch_sizes:                
        #         self._allocate_tensor((batch_size, hidden_dim), torch.bfloat16, 2)
                
        # # preallocate router_logits tensors
        # for router_logits_num in router_logits_num:
        #     for batch_size in batch_sizes:
        #         self._allocate_tensor((batch_size, router_logits_num), torch.bfloat16, 2)
    
    def _allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype, count: int = 1):        
        key = (shape, dtype)
        if key not in self.free_tensors:
            self.free_tensors[key] = []
        
        # calculate the number of bytes of a single tensor
        element_size = torch.tensor([], dtype=dtype).element_size()
        tensor_bytes = torch.tensor(shape).prod().item() * element_size
        
        # check if the total memory limit is exceeded
        for _ in range(count):
            if self.allocated_bytes + tensor_bytes <= self.total_size_mb * 1024 * 1024:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self.free_tensors[key].append(tensor)
                self.allocated_bytes += tensor_bytes
            else:
                logger.warning(f"The memory pool has reached the maximum size of {self.total_size_mb}MB")
                break
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Get a tensor of the specified shape and type, if the pool is empty, create a new one"""
        with self.lock:
            key = (shape, dtype)
            
            # if there is no available tensor of this shape, allocate a new one
            if key not in self.free_tensors or not self.free_tensors[key]:
                self._allocate_tensor(shape, dtype, 1)
                
                # if after allocation, there is still no available tensor, create a temporary tensor directly
                if not self.free_tensors[key]:
                    logger.warning(f"The memory pool is full, create a temporary tensor: {shape}, {dtype}")
                    return torch.zeros(shape, dtype=dtype, device=self.device)
            
            # get a tensor from the pool
            tensor = self.free_tensors[key].pop()
            # ensure the tensor is zeroed
            tensor.zero_()
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):        
        if tensor.device.type != self.device:
            return  # ignore tensors not on the specified device
            
        with self.lock:
            key = (tuple(tensor.shape), tensor.dtype)
            
            # if the list of this shape does not exist, create one
            if key not in self.free_tensors:
                self.free_tensors[key] = []
                
            # return the tensor
            self.free_tensors[key].append(tensor)
    
    def clear(self):        
        with self.lock:
            self.free_tensors.clear()
            self.allocated_bytes = 0
            # force garbage collection
            torch.cuda.empty_cache()
            
    def get_stats(self) -> Dict:
        with self.lock:
            stats = {
                "allocated_mb": self.allocated_bytes / (1024 * 1024),
                "total_mb": self.total_size_mb,
                "usage_percent": (self.allocated_bytes / (self.total_size_mb * 1024 * 1024)) * 100,
                "tensor_types": len(self.free_tensors),
                "tensors_available": sum(len(tensors) for tensors in self.free_tensors.values())
            }
            return stats 