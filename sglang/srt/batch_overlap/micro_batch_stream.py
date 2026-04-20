
import torch


class MBOStreamManager:
    def __init__(self, num_micro_batches: int):
        self.streams = [torch.cuda.Stream() for _ in range(num_micro_batches)]
    
    def get_stream(self, micro_batch_idx: int) -> torch.cuda.Stream:
        return self.streams[micro_batch_idx]
