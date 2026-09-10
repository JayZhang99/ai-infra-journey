from __future__ import annotations

from copy import deepcopy
import pytest 

from python.serving_scheduler import(
    BlockAllocator,
    CapacityError,
    ServingScheduler,
    SchedulerConfig,
    Sequence,
    IterationPlan,
    RequestState,
)

def test_allocate_failure_is_atomic():
    allocator = BlockAllocator(2)
    before = allocator.snapshot()

    with pytest.raises(CapacityError):
        allocator.allocate(3)
    
    assert allocator.snapshot() == before
    allocator.validate_invariants()


def test_allocate_zero():
    allocator = BlockAllocator(2)
    before = allocator.snapshot()
    allocator.allocate(0)        
    
    assert allocator.snapshot() == before
    allocator.validate_invariants()

def test_restore():
    allocator = BlockAllocator(2)

    allocator.allocate(2)
    assert allocator.free_count == 0

    allocator.release([0,1])
    assert allocator.free_count == 2

def test_retain_add_refcount():
    allocator = BlockAllocator(2)

    allocator.allocate(2)

    allocator.retain([1])

    assert allocator.refcount(0) == 1
    assert allocator.refcount(1) == 2

def test_release_refcount_zero():
    allocator = BlockAllocator(2)

    allocator.allocate(2)

    allocator.retain([1])

    allocator.release([0,1])
    
    assert allocator.free_count == 1

def test_FCFS():
    allocator = BlockAllocator(total_blocks=64)
    config = SchedulerConfig(
    token_budget=16,
    max_sequences=4,
    block_size=4,
    prefill_chunk_size=8,
    policy="fcfs",
    )

    sequences = [
        Sequence(
            request_id="late-decode",
            arrival_index=20,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
        Sequence(
            request_id="early-prefill",
            arrival_index=5,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.PREFILL,
        ),
        Sequence(
            request_id="middle-waiting",
            arrival_index=10,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.WAITING,
        ),
    ]
    scheduler = ServingScheduler(config, allocator)
    ordered = scheduler.ordered(sequences)

    actual_ids = [
        seq.request_id
        for seq in ordered
    ]

    assert actual_ids == [
        "early-prefill",
        "middle-waiting",
        "late-decode",
    ]


def test_Decode_first():
    allocator = BlockAllocator(total_blocks=64)

    config = SchedulerConfig(
    token_budget=16,
    max_sequences=4,
    block_size=4,
    prefill_chunk_size=8,
    policy="decode_first",
    )

    sequences = [
        Sequence(
            request_id="late-decode",
            arrival_index=20,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
        Sequence(
            request_id="early-prefill",
            arrival_index=5,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.PREFILL,
        ),
        Sequence(
            request_id="middle-waiting",
            arrival_index=10,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.WAITING,
        ),
    ]
    scheduler = ServingScheduler(config, allocator)
    ordered = scheduler.ordered(sequences)

    actual_ids = [
        seq.request_id
        for seq in ordered
    ]

    assert actual_ids == [
        "late-decode",
        "early-prefill",
        "middle-waiting",
    ]

def test_token_count_less_budget():
    allocator = BlockAllocator(total_blocks=64)

    config = SchedulerConfig(
    token_budget=16,
    max_sequences=4,
    block_size=4,
    prefill_chunk_size=8,
    policy="decode_first",
    )

    sequences = [
        Sequence(
            request_id="late-decode",
            arrival_index=20,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
        Sequence(
            request_id="early-prefill",
            arrival_index=5,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.PREFILL,
        ),
        Sequence(
            request_id="middle-waiting",
            arrival_index=10,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.WAITING,
        ),
    ]
    scheduler = ServingScheduler(config, allocator)
    iteration = scheduler.schedule_iteration(sequences)
    assert iteration.token_count <= config.token_budget
    assert iteration.token_count == 16
    assert list(iteration.scheduled_tokens) == [
        "late-decode",
        "early-prefill",
        "middle-waiting",
    ]

def test_orded_less_max_sequences():
    allocator = BlockAllocator(total_blocks=64)

    config = SchedulerConfig(
    token_budget=16,
    max_sequences=2,
    block_size=4,
    prefill_chunk_size=8,
    policy="decode_first",
    )

    sequences = [
        Sequence(
            request_id="late-decode",
            arrival_index=20,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
        Sequence(
            request_id="early-prefill",
            arrival_index=5,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.PREFILL,
        ),
        Sequence(
            request_id="middle-waiting",
            arrival_index=10,
            prompt_remaining=8,
            output_remaining=4,
            state=RequestState.WAITING,
        ),
    ]
    scheduler = ServingScheduler(config, allocator)
    plan = scheduler.schedule_iteration(sequences)
    assert len(plan.scheduled_tokens) == 2
    assert list(plan.scheduled_tokens) == [
        "late-decode",
        "early-prefill"
    ]
    
def test_scheduler_failure_is_atomic():
    # 只有一个 KV block，但两个 Decode 请求各需要一个新 block。
    allocator = BlockAllocator(total_blocks=1)

    config = SchedulerConfig(
        token_budget=2,
        max_sequences=2,
        block_size=4,
        prefill_chunk_size=4,
        policy="fcfs",
    )

    sequences = [
        Sequence(
            request_id="first",
            arrival_index=0,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
        Sequence(
            request_id="second",
            arrival_index=1,
            prompt_remaining=0,
            output_remaining=4,
            state=RequestState.DECODE,
        ),
    ]

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    before_allocator = allocator.snapshot()
    before_sequences = deepcopy(sequences)

    with pytest.raises(CapacityError):
        scheduler.schedule_iteration(sequences)

    # 第一个请求临时预留的 block 必须被释放。
    assert allocator.snapshot() == before_allocator

    # prepare 失败不能修改任何 Sequence。
    assert sequences == before_sequences

    allocator.validate_invariants()

def test_rollback_releases_only_reserved_blocks():
    allocator = BlockAllocator(total_blocks=4)

    # 模拟调度开始前已经存在的分配。
    existing_blocks = allocator.allocate(1)
    existing_block = existing_blocks[0]

    config = SchedulerConfig(
        token_budget=8,
        max_sequences=1,
        block_size=4,
        prefill_chunk_size=8,
        policy="fcfs",
    )

    sequence = Sequence(
        request_id="prefill-request",
        arrival_index=0,
        prompt_remaining=8,
        output_remaining=4,
        state=RequestState.PREFILL,
    )

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    before_allocator = allocator.snapshot()
    before_sequence = deepcopy(sequence)

    # Prepare 阶段为 8 个 token 预留两个 KV blocks。
    plan = scheduler.schedule_iteration([sequence])

    assert plan.closed is False
    assert plan.token_count == 8

    assert len(
        plan.reserved_blocks["prefill-request"]
    ) == 2

    # schedule_iteration 只能预留资源，不能修改 Sequence。
    assert sequence == before_sequence

    # 原有 block 加上本轮两个预留，当前共占用三个。
    assert allocator.free_count == 1
    assert allocator.refcount(existing_block) == 1

    scheduler.rollback(plan)

    # Rollback 后 Plan 被关闭。
    assert plan.closed is True

    # Sequence 从始至终都不能变化。
    assert sequence == before_sequence

    before_free, before_refcounts = before_allocator
    after_free, after_refcounts = allocator.snapshot()

    # Free deque 的顺序可以变化，因此比较集合。
    assert set(after_free) == set(before_free)

    # 所有 block 的所有权必须恢复。
    assert after_refcounts == before_refcounts

    # 调度前已经存在的 block 仍然属于原持有者。
    assert allocator.refcount(existing_block) == 1

    allocator.validate_invariants()


def test_commit_after_success():
    allocator = BlockAllocator(total_blocks=4)
    config = SchedulerConfig(
        token_budget=8,
        max_sequences=1,
        block_size=4,
        prefill_chunk_size=8,
        policy="fcfs",
    )

    sequence = Sequence(
        request_id="prefill-request",
        arrival_index=0,
        prompt_remaining=8,
        output_remaining=4,
        state=RequestState.PREFILL,
    )
    expected_prompt = 0
    expected_kv = 8
    expected_state = RequestState.DECODE

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    plan = scheduler.schedule_iteration([sequence])
    scheduler.commit(plan, {sequence.request_id: sequence})

    assert sequence.prompt_remaining == expected_prompt
    assert sequence.kv_tokens == expected_kv
    assert sequence.state is expected_state

def test_share_partial_prefix_reuses_two_blocks():
    allocator = BlockAllocator(total_blocks=4)

    config = SchedulerConfig(
        token_budget=8,
        max_sequences=2,
        block_size=4,
        prefill_chunk_size=8,
        policy="fcfs",
    )

    source_blocks = allocator.allocate(2)

    source = Sequence(
        request_id="source",
        arrival_index=0,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
        kv_tokens=8,
        block_table=list(source_blocks),
    )

    target = Sequence(
        request_id="target",
        arrival_index=1,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
    )

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    scheduler.share_prefix(
        source=source,
        target=target,
        shared_tokens=6,
    )

    assert target.kv_tokens == 6
    assert target.block_table == source_blocks

    assert allocator.refcount(source_blocks[0]) == 2
    assert allocator.refcount(source_blocks[1]) == 2

    allocator.validate_invariants()

def test_cow_success():
    allocator = BlockAllocator(total_blocks=4)

    config = SchedulerConfig(
        token_budget=8,
        max_sequences=2,
        block_size=4,
        prefill_chunk_size=8,
        policy="fcfs",
    )

    source_blocks = allocator.allocate(2)

    source = Sequence(
        request_id="source",
        arrival_index=0,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
        kv_tokens=8,
        block_table=list(source_blocks),
    )

    target = Sequence(
        request_id="target",
        arrival_index=1,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
    )

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    scheduler.share_prefix(
        source=source,
        target=target,
        shared_tokens=6,
    )

    old = source.block_table[-1]
    new = scheduler.copy_on_write_tail(
        target,
        lambda old_block, new_block: None,
    )

    assert source.block_table[-1] == old
    assert target.block_table[-1] == new

def test_cow_copy_failure_is_atomic():
    allocator = BlockAllocator(total_blocks=4)

    config = SchedulerConfig(
        token_budget=1,
        max_sequences=1,
        block_size=4,
        prefill_chunk_size=4,
        policy="fcfs",
    )

    # Source有6个KV token：
    # block 0 = [4/4]
    # block 1 = [2/4]，尾块未写满。
    source_blocks = allocator.allocate(2)

    source = Sequence(
        request_id="source",
        arrival_index=0,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
        kv_tokens=6,
        block_table=list(source_blocks),
    )

    target = Sequence(
        request_id="target",
        arrival_index=1,
        prompt_remaining=0,
        output_remaining=4,
        state=RequestState.DECODE,
    )

    scheduler = ServingScheduler(
        config,
        allocator,
    )

    # Target共享Source的两个物理block。
    scheduler.share_prefix(
        source=source,
        target=target,
        shared_tokens=6,
    )

    shared_head = source_blocks[0]
    shared_tail = source_blocks[1]

    assert source.block_table == [
        shared_head,
        shared_tail,
    ]

    assert target.block_table == [
        shared_head,
        shared_tail,
    ]

    assert allocator.refcount(shared_head) == 2
    assert allocator.refcount(shared_tail) == 2

    source_before = deepcopy(source)
    target_before = deepcopy(target)

    before_free_ids, before_refcounts = (
        allocator.snapshot()
    )

    before_free_count = allocator.free_count

    copy_calls: list[tuple[int, int]] = []

    def fail_copy(
        old_block: int,
        new_block: int,
    ) -> None:
        copy_calls.append(
            (old_block, new_block)
        )

        # 复制发生时，映射还不能提前改变。
        assert target.block_table[-1] == old_block

        # 旧尾块仍然由Source和Target共同持有。
        assert allocator.refcount(old_block) == 2

        # 新block已经临时分配。
        assert allocator.refcount(new_block) == 1

        raise RuntimeError(
            "simulated CUDA copy failure"
        )

    with pytest.raises(
        RuntimeError,
        match="simulated CUDA copy failure",
    ):
        scheduler.copy_on_write_tail(
            target,
            fail_copy,
        )

    # 回调确实执行了一次。
    assert len(copy_calls) == 1

    copied_old, attempted_new = copy_calls[0]

    assert copied_old == shared_tail
    assert attempted_new != shared_tail

    # Source完全没有变化。
    assert source == source_before

    # Target映射不能切换到复制失败的新block。
    assert target == target_before
    assert target.block_table[-1] == shared_tail

    # 旧尾块仍由Source和Target共同持有。
    assert allocator.refcount(shared_tail) == 2

    # 临时申请的新block已经释放。
    assert allocator.refcount(attempted_new) == 0

    after_free_ids, after_refcounts = (
        allocator.snapshot()
    )

    # 可用容量恢复。
    assert allocator.free_count == before_free_count

    # 可用block集合恢复。
    assert set(after_free_ids) == set(before_free_ids)

    # 所有物理block的refcount恢复。
    assert after_refcounts == before_refcounts

    # 失败时尝试使用的新block重新回到free pool。
    assert attempted_new in after_free_ids

    allocator.validate_invariants()