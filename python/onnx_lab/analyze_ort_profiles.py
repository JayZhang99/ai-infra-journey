from __future__ import annotations 

import json
from pathlib import Path
import argparse
from typing import Any, Sequence


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('unsupported schema_version')
    return data


def validate_summary(data: dict[str, Any]) -> None:
    cfg = data['configuration']
    env = data['environment']
    steps = cfg['steps']

    if env['selected_providers'] != ['CPUExecutionProvider']:
        raise ValueError('unexpected providers')
    if data['session_event_counts']['model_run'] != steps:
        raise ValueError('model_run count mismatch')
    if sum(data['op_counts'].values()) != data['node_event_count']:
        raise ValueError('op count mismatch')
    if sum(data['provider_counts'].values()) != data['node_event_count']:
        raise ValueError('provider count mismatch')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    if data["node_event_count"] % steps != 0:
        raise ValueError(
            "node_event_count is not divisible by steps"
        )

    for op_name, count in data["op_counts"].items():
        if count % steps != 0:
            raise ValueError(
                f"op count for {op_name} "
                "is not divisible by steps"
            )
    
def structure_row(data: dict[str, Any]) -> dict[str, Any]:
    validate_summary(data)

    cfg = data["configuration"]
    steps = cfg["steps"]
    counts = data["op_counts"]
    durations = data.get("op_total_dur_us", {})
    optimized = (
        data.get("artifacts", {})
        .get("optimized_model", {})
    )

    return {
        "level": cfg["optimization_level"],

        # 静态图每轮节点数应一致，因此使用整数除法。
        "nodes_per_run": data["node_event_count"] // steps,

        "ops_per_run": {
            name: count // steps
            for name, count in counts.items()
        },

        # 明确这是带 Profiler 时采集的算子时间，
        # 不能直接当作端到端推理延迟。
        "profile_op_dur_us_per_run": {
            name: float(duration) / steps
            for name, duration in durations.items()
        },

        "selected_providers":
            data["environment"]["selected_providers"],

        "session_creation_ms":
            float(data["session"]["creation_ms"]),

        "optimized_model_size":
            optimized.get("size"),

        "optimized_model_sha256":
            optimized.get("sha256"),
    }

def detect_quickgelu_boundary(
    by_level: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    basic = by_level["basic"]["ops_per_run"]
    extended = by_level["extended"]["ops_per_run"]

    observed = (
        basic.get("Sigmoid", 0) > 0
        and basic.get("QuickGelu", 0) == 0
        and extended.get("Sigmoid", 0) == 0
        and extended.get("QuickGelu", 0) > 0
    )

    return {
        "observed": observed,
        "observed_from": (
            "extended" if observed else None
        ),
        "evidence": {
            "basic": {
                "Sigmoid": basic.get("Sigmoid", 0),
                "QuickGelu": basic.get("QuickGelu", 0),
                "Mul": basic.get("Mul", 0),
            },
            "extended": {
                "Sigmoid": extended.get("Sigmoid", 0),
                "QuickGelu": extended.get("QuickGelu", 0),
                "Mul": extended.get("Mul", 0),
            },
        },
    }


LEVEL_ORDER = (
    "disable",
    "basic",
    "extended",
    "all",
)


def comparison_signature(
    data: dict[str, Any],
) -> dict[str, Any]:
    cfg = data["configuration"]
    model = data.get("artifacts", {}).get("model", {})

    return {
        "model_sha256": model.get("sha256"),
        "shape": cfg.get("shape"),
        "dtype": cfg.get("dtype"),
        "steps": cfg.get("steps"),
        "threads": cfg.get("threads"),
        "selected_providers":
            data["environment"]["selected_providers"],
    }


def validate_compatible(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = comparison_signature(items[0])

    for data in items[1:]:
        current = comparison_signature(data)

        if current != baseline:
            level = data["configuration"][
                "optimization_level"
            ]
            changed = [
                key
                for key in baseline
                if baseline[key] != current[key]
            ]
            raise ValueError(
                f"incompatible summary for {level}: "
                + ", ".join(changed)
            )

    return baseline


def build_comparison(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(items) != len(LEVEL_ORDER):
        raise ValueError(
            "expected exactly four optimization levels"
        )

    for data in items:
        validate_summary(data)

    contract = validate_compatible(items)
    rows = [structure_row(data) for data in items]

    row_by_level: dict[str, dict[str, Any]] = {}

    for row in rows:
        level = row["level"]

        if level in row_by_level:
            raise ValueError(
                f"duplicate optimization level: {level}"
            )

        row_by_level[level] = row

    missing = [
        level
        for level in LEVEL_ORDER
        if level not in row_by_level
    ]
    if missing:
        raise ValueError(
            "missing optimization levels: "
            + ", ".join(missing)
        )

    # 强制稳定输出顺序。
    by_level = {
        level: row_by_level[level]
        for level in LEVEL_ORDER
    }

    return {
        "schema_version": 1,
        "experiment":
            "ort_optimization_level_comparison",
        "comparison_contract": contract,
        "levels": by_level,
        "fusion_boundary": {
            "quickgelu":
                detect_quickgelu_boundary(by_level),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four ORT profile summaries"
        )
    )
    parser.add_argument(
        "--inputs",
        nargs=4,
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    items = [
        load_summary(path)
        for path in args.inputs
    ]
    result = build_comparison(items)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())