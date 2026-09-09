from __future__ import annotations

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


def tset_allocate_zero():
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
