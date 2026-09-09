from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Callable, Mapping

class CapacityError(RuntimeError):
    """KV Cache block capacity is insufficient."""

class RequestState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE  = "decode"
    FINISHED = "finished"
    ABORTED = "aborted"

@dataclass
class Sequence:
    request_id: str
    arrival_index: int
    prompt_remaining: int
    output_remaining: int

    state: RequestState = RequestState.WAITING
    kv_tokens: int = 0
    block_table: list[int] = field(default_factory=list)

@dataclass(frozen=True)
class SchedulerConfig:
    token_budget: int
    max_sequences: int
    block_size: int
    prefill_chunk_size: int
    policy: str

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")

        if self.max_sequences <= 0:
            raise ValueError("max_sequences must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.prefill_chunk_size <= 0:
            raise ValueError(
                "prefill_chunk_size must be positive"
            )

        if self.policy not in {"fcfs", "decode_first"}:
            raise ValueError(
                "policy must be fcfs or decode_first"
            )

class BlockAllocator:
    def __init__(self, total_blocks: int) -> None:
        if total_blocks < 0:
            raise ValueError(
                "total_blocks must be non-negative"
            )

        self.total_blocks = total_blocks
        self._free = deque(range(total_blocks))
        self._refcount = [0] * total_blocks
    @property
    def free_count(self) -> int:
        return len(self._free)

    def refcount(self, block_id: int) -> int:
        self._validate_id(block_id)
        return self._refcount[block_id]
    
    def snapshot(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(self._free), tuple(self._refcount)

    def allocate(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError(
                "count must be non-negative"
            )

        # 必须在修改状态前检查容量，保证失败原子性。
        if count > self.free_count:
            raise CapacityError(
                "not enough free KV blocks"
            )

        block_ids = [
            self._free.popleft()
            for _ in range(count)
        ]

        for block_id in block_ids:
            self._refcount[block_id] = 1

        return block_ids

    def retain(self, block_ids: list[int]) -> None:
        self._validate_owned_unique(block_ids)

        for block_id in block_ids:
            self._refcount[block_id] += 1
        
    
    def release(self, block_ids: list[int]) -> None:
        # 先验证全部 block，再开始修改。
        self._validate_owned_unique(block_ids)

        for block_id in block_ids:
            self._refcount[block_id] -= 1

            if self._refcount[block_id] == 0:
                self._free.append(block_id)
    
    def validate_invariants(self) -> None:
        free_list = list(self._free)
        free_set = set(free_list)
        legal_ids = set(range(self.total_blocks))

        # 1. free pool 无重复。
        assert len(free_list) == len(free_set)

        # 2. block id 合法。
        assert free_set <= legal_ids

        # 3. refcount 不为负。
        assert all(
            count >= 0
            for count in self._refcount
        )

        allocated = {
            block_id
            for block_id, count
            in enumerate(self._refcount)
            if count > 0
        }

        zero_ref = {
            block_id
            for block_id, count
            in enumerate(self._refcount)
            if count == 0
        }

        # 4. free block 的 refcount 必须为 0。
        assert free_set == zero_ref

        # 5. free 和 allocated 不相交。
        assert free_set.isdisjoint(allocated)

        # 6. free + allocated == total。
        assert free_set | allocated == legal_ids


    def _validate_id(self, block_id: int) -> None:
        if not 0 <= block_id < self.total_blocks:
            raise IndexError(
                f"illegal block id: {block_id}"
            )

    def _validate_owned_unique(
        self,
        block_ids: list[int],
    ) -> None:
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("duplicate block id")

        for block_id in block_ids:
            self._validate_id(block_id)

            if self._refcount[block_id] == 0:
                raise ValueError(
                    f"block {block_id} is not allocated"
                )
    
@dataclass
class IterationPlan:
    decode_ids: list[str]
    prefill_tokens: dict[str, int]

    # 每个请求本轮计算的 token 数。
    scheduled_tokens: dict[str, int]

    # prepare 阶段预留，commit 时加入 block_table。
    reserved_blocks: dict[str, list[int]]

    token_count: int
    policy_name: str

    closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

class ServingScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        allocator: BlockAllocator,
    ) -> None:
        self.config = config
        self.allocator = allocator

    def ordered(
        self,
        sequences: list[Sequence],
    ) -> list[Sequence]:
        live = [
            seq
            for seq in sequences
            if seq.state not in {
                RequestState.FINISHED,
                RequestState.ABORTED,
            }
        ]

        # arrival_index 代表进入系统的先后顺序。
        live.sort(
            key=lambda seq: seq.arrival_index
        )

        if self.config.policy == "fcfs":
            return live

        decode = [
            seq
            for seq in live
            if seq.state is RequestState.DECODE
        ]

        others = [
            seq
            for seq in live
            if seq.state is not RequestState.DECODE
        ]

        return decode + others

    def schedule_iteration(
        self,
        sequences: list[Sequence],
    ) -> IterationPlan:
        """
        prepare 阶段。

        可以预留 KV blocks，但不能修改 Sequence。
        """

        self._validate_sequences(sequences)

        remaining_budget = self.config.token_budget

        scheduled: dict[str, int] = {}
        prefill: dict[str, int] = {}
        decode: list[str] = []
        reserved: dict[str, list[int]] = {}

        try:
            for seq in self.ordered(sequences):
                # sequence slots 约束。
                if (
                    len(scheduled)
                    == self.config.max_sequences
                ):
                    break

                token_count = self._tokens_for(
                    seq,
                    remaining_budget,
                )

                if token_count == 0:
                    continue

                self._require_writable_tail(seq)

                block_count = self._new_block_count(
                    seq,
                    token_count,
                )

                # KV block 约束。
                block_ids = self.allocator.allocate(
                    block_count
                )

                request_id = seq.request_id

                reserved[request_id] = block_ids
                scheduled[request_id] = token_count

                if seq.state is RequestState.DECODE:
                    decode.append(request_id)
                else:
                    prefill[request_id] = token_count

                remaining_budget -= token_count

                if remaining_budget == 0:
                    break

        except Exception:
            # prepare 中任何请求失败，都释放本轮已预留资源。
            for block_ids in reserved.values():
                self.allocator.release(block_ids)

            raise

        return IterationPlan(
            decode_ids=decode,
            prefill_tokens=prefill,
            scheduled_tokens=scheduled,
            reserved_blocks=reserved,
            token_count=sum(scheduled.values()),
            policy_name=self.config.policy,
        )

    def commit(
        self,
        plan: IterationPlan,
        by_id: Mapping[str, Sequence],
    ) -> None:
        """
        GPU 执行成功后，才修改 Sequence。
        """

        self._require_open(plan)

        missing = (
            set(plan.scheduled_tokens)
            - set(by_id)
        )

        if missing:
            raise KeyError(
                f"missing sequences: {sorted(missing)}"
            )

        for request_id, token_count in (
            plan.scheduled_tokens.items()
        ):
            seq = by_id[request_id]

            seq.block_table.extend(
                plan.reserved_blocks[request_id]
            )

            seq.kv_tokens += token_count

            if request_id in plan.prefill_tokens:
                seq.prompt_remaining -= token_count

                if seq.prompt_remaining > 0:
                    seq.state = RequestState.PREFILL
                elif seq.output_remaining > 0:
                    seq.state = RequestState.DECODE
                else:
                    seq.state = RequestState.FINISHED

            else:
                seq.output_remaining -= token_count

                if seq.output_remaining == 0:
                    seq.state = RequestState.FINISHED
                else:
                    seq.state = RequestState.DECODE

        plan.closed = True
        self.allocator.validate_invariants()

    def rollback(
        self,
        plan: IterationPlan,
    ) -> None:
        """
        GPU 执行失败时，仅释放本轮预留的 blocks。
        """

        self._require_open(plan)

        for block_ids in plan.reserved_blocks.values():
            self.allocator.release(block_ids)

        plan.closed = True
        self.allocator.validate_invariants()

    def reclaim_finished(
        self,
        sequences: list[Sequence],
    ) -> None:
        """
        在下一轮 iteration 开始时回收已结束请求。
        """

        for seq in sequences:
            if seq.state not in {
                RequestState.FINISHED,
                RequestState.ABORTED,
            }:
                continue

            if seq.block_table:
                self.allocator.release(
                    seq.block_table
                )
                seq.block_table.clear()

            seq.kv_tokens = 0

    def share_prefix(
        self,
        source: Sequence,
        target: Sequence,
        shared_tokens: int,
    ) -> None:
        """
        让 target 共享 source 的 KV Prefix。
        """

        if target.block_table or target.kv_tokens:
            raise ValueError(
                "target already owns KV blocks"
            )

        if not 0 <= shared_tokens <= source.kv_tokens:
            raise ValueError(
                "invalid shared_tokens"
            )

        block_count = ceil(
            shared_tokens
            / self.config.block_size
        )

        prefix = source.block_table[:block_count]

        # 每多一个 owner，refcount 加一。
        self.allocator.retain(prefix)

        target.block_table = list(prefix)
        target.kv_tokens = shared_tokens

    def copy_on_write_tail(
        self,
        seq: Sequence,
        copy_valid_kv: Callable[[int, int], None],
    ) -> int | None:
        """
        如果最后一个未写满 block 被共享，
        创建私人副本后再允许继续写入。
        """

        if not seq.block_table:
            return None

        old_block = seq.block_table[-1]

        tail_is_partial = (
            seq.kv_tokens
            % self.config.block_size
            != 0
        )

        tail_is_shared = (
            self.allocator.refcount(old_block)
            > 1
        )

        if not tail_is_partial or not tail_is_shared:
            return old_block

        new_block = self.allocator.allocate(1)[0]

        try:
            # 在 CUDA 实现中，这里复制有效 KV 内容。
            copy_valid_kv(
                old_block,
                new_block,
            )
        except Exception:
            # 复制失败不改变 block_table。
            self.allocator.release([new_block])
            raise

        # 复制成功后才替换映射。
        seq.block_table[-1] = new_block

        # 当前 Sequence 不再引用旧 block。
        self.allocator.release([old_block])

        return new_block

    def _tokens_for(
        self,
        seq: Sequence,
        budget: int,
    ) -> int:
        if budget <= 0:
            return 0

        if seq.state in {
            RequestState.WAITING,
            RequestState.PREFILL,
        }:
            return min(
                seq.prompt_remaining,
                self.config.prefill_chunk_size,
                budget,
            )

        if (
            seq.state is RequestState.DECODE
            and seq.output_remaining > 0
        ):
            # 每条 Decode Sequence 本轮生成一个 token。
            return 1

        return 0

    def _new_block_count(
        self,
        seq: Sequence,
        scheduled_tokens: int,
    ) -> int:
        required = ceil(
            (
                seq.kv_tokens
                + scheduled_tokens
            )
            / self.config.block_size
        )

        return max(
            0,
            required - len(seq.block_table),
        )

    def _require_writable_tail(
        self,
        seq: Sequence,
    ) -> None:
        if not seq.block_table:
            return

        partial = (
            seq.kv_tokens
            % self.config.block_size
            != 0
        )

        shared = (
            self.allocator.refcount(
                seq.block_table[-1]
            )
            > 1
        )

        if partial and shared:
            raise RuntimeError(
                "shared partial tail requires COW"
            )

    def _validate_sequences(
        self,
        sequences: list[Sequence],
    ) -> None:
        request_ids = [
            seq.request_id
            for seq in sequences
        ]

        if len(request_ids) != len(set(request_ids)):
            raise ValueError(
                "duplicate request_id"
            )

        for seq in sequences:
            if (
                seq.prompt_remaining < 0
                or seq.output_remaining < 0
                or seq.kv_tokens < 0
            ):
                raise ValueError(
                    "token counts cannot be negative"
                )

            expected_blocks = ceil(
                seq.kv_tokens
                / self.config.block_size
            )

            if len(seq.block_table) != expected_blocks:
                raise ValueError(
                    "block_table does not match kv_tokens"
                )

    @staticmethod
    def _require_open(
        plan: IterationPlan,
    ) -> None:
        if plan.closed:
            raise RuntimeError(
                "iteration plan is already closed"
            )    

