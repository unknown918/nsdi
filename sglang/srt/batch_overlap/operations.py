from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Sequence, Union

from sglang.srt.configs.logger_config import configure_logger
logger = configure_logger(__name__)

import torch
import asyncio

class YieldOperation:
    pass

@dataclass
class ExecutionOperation:
    debug_name: str
    fn: Callable

Operation = Union[YieldOperation, ExecutionOperation, Callable]
Stage = List[ExecutionOperation]

async def execute_overlapped_operations(
    inputs_arr: Sequence,
    operations_arr: Sequence,
    delta_stage: int,
    num_micro_batch: int,
    mbo_stream_manager=None,
) -> Sequence:
    # logger.Model(f"[MBO] start to overlap execute: input len {len(inputs_arr)}")
    executors = [
        {
            "executor": _AsyncStageExecutor(
                f"micro_batch_{i}",
                _convert_operations_to_stages(operations_arr[i]),
                inputs=inputs_arr[i],
                batch_id=i,
                mbo_stream_manager=mbo_stream_manager,
            ),
            "offset": i * delta_stage, # assume all micro-batches have same delta_stage
        }
        for i in range(num_micro_batch)
    ]

    t = 0
    while not all(ex["executor"].done for ex in executors):
        # with torch.profiler.record_function(f"t={t}"):
        with torch.cuda.nvtx.range(f"step_{t}"):
            tasks = []
            for ex in executors:
                if t >= ex["offset"] and not ex["executor"].done:
                    # logger.Model(f"[MBO] step[{t}], run executor {ex["executor"]._debug_name}, batch_id {ex["executor"]._stage_state.get_batch_id()}")
                    # with torch.profiler.record_function(f"stage_{(t-ex["offset"])%4}_ex_{ex["offset"]}"):
                    tasks.append(ex["executor"].next())
            await asyncio.gather(*tasks)
        t += 1

    # tasks = []
    # for ex in executors:
    #     tasks.append(ex["executor"].run())
    # await asyncio.gather(*tasks)

    assert all(ex["executor"].done for ex in executors)
    return [ex["executor"].output for ex in executors]

class _AsyncStageExecutor:
    def __init__(self, debug_name: str, stages: List[Stage], inputs: dict, batch_id: int, mbo_stream_manager=None):
        self._debug_name = debug_name
        self._stages = stages
        self._index = 0
        self._stage_state = _StateDict(batch_id)
        self._stage_output = inputs
        self._cuda_stream = mbo_stream_manager.get_stream(batch_id)
    
    async def next(self):
        assert not self.done

        stage = self._stages[self._index]

        #with _annotate_region(debug_name=f"{self._debug_name}{self._index}"):
        with torch.cuda.stream(self._cuda_stream):
            for op in stage:
                # with _annotate_region(debug_name=op.debug_name):
                self._stage_output = await op.fn(
                    state=self._stage_state,
                    **(
                        self._stage_output if self._stage_output is not None else {}
                    ),
                )

        self._index += 1

    async def run(self):
        while not self.done:
            await self.next()

    @property
    def output(self):
        assert self.done
        return self._stage_output

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)


class _StateDict:
    def __init__(self, batch_id: int):
        self._data = {}
        super().__setattr__("batch_id", batch_id) # bypass

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
            return
        assert (
            key not in self._data
        ), f"`{key}` already exist, are you sure you want to override it?"
        self._data[key] = value

    def __getattr__(self, item):
        return self._data[item]

    def __delattr__(self, item):
        del self._data[item]

    def pop(self, item):
        return self._data.pop(item)

    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)

    def get(self, item):
        return self._data.get(item)

    def clear(self, expect_keys: Sequence[str]):
        if set(self._data.keys()) != set(expect_keys):
            raise Exception(
                f"Unexpected keys when clearing. This may indicate you do not release memory early enough but leave it to here. {list(self._data.keys())=} {expect_keys=}"
            )

        self._data.clear()

    def get_batch_id(self):
        return super().__getattribute__("batch_id")

def _convert_operations_to_stages(operations: List[Operation]) -> List[Stage]:
    operations = _decorate_operations(operations)
    operation_chunks = list(
        _chunk_by_separator(operations, lambda op: isinstance(op, YieldOperation))
    )
    assert all(len(chunk) > 0 for chunk in operation_chunks)
    return operation_chunks


def _chunk_by_separator(
    items: List[Any], is_separator: Callable[[Any], bool]
) -> Generator[List[Any], None, None]:
    pending_items = []
    for item in items:
        if is_separator(item):
            yield pending_items
            pending_items = []
        else:
            pending_items.append(item)
    if len(pending_items) > 0:
        yield pending_items


def _decorate_operations(operations: List[Operation], debug_name_prefix: str = ""):
    return [_decorate_operation(op, debug_name_prefix) for op in operations]


def _decorate_operation(operation: Operation, debug_name_prefix: str):
    if isinstance(operation, YieldOperation):
        return operation
    return ExecutionOperation(
        debug_name=debug_name_prefix
        + getattr(operation, "__name__", "unknown").replace("op_", ""),
        fn=operation,
    )
