# SPDX-License-Identifier: Apache-2.0
"""Require identical greedy token output from two benchmark result files."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import json


def successful_outputs(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = [request for request in payload["requests"] if request["error"]]
    if failures:
        raise ValueError(f"{path} contains {len(failures)} failed requests")
    return {
        request["request_id"]: request["generated_token_ids"]
        for request in payload["requests"]
    }


def compare(left: Path, right: Path) -> None:
    left_outputs = successful_outputs(left)
    right_outputs = successful_outputs(right)
    if left_outputs.keys() != right_outputs.keys():
        missing_left = sorted(right_outputs.keys() - left_outputs.keys())
        missing_right = sorted(left_outputs.keys() - right_outputs.keys())
        raise ValueError(
            f"request IDs differ: missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )
    mismatches = [
        request_id
        for request_id in left_outputs
        if left_outputs[request_id] != right_outputs[request_id]
    ]
    if mismatches:
        raise ValueError(f"generated token IDs differ for requests: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    compare(args.left, args.right)
    print(f"identical greedy outputs: {args.left} == {args.right}")


if __name__ == "__main__":
    main()
