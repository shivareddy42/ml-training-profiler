"""
Compare benchmark results side by side.

Usage:
    python -m src.compare --baseline benchmarks/baseline.json --optimized benchmarks/all_combined.json
"""

import argparse
import json


def compare(baseline_path: str, optimized_path: str):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    b_tp = baseline["avg_throughput_img_per_sec"]
    o_tp = optimized["avg_throughput_img_per_sec"]
    speedup = o_tp / max(b_tp, 1)

    b_mem = baseline.get("peak_memory_mb", 0)
    o_mem = optimized.get("peak_memory_mb", 0)
    mem_change = ((o_mem - b_mem) / max(b_mem, 1)) * 100

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
    print("=" * 66)
    print(f"{'Throughput (img/s)':<30} {b_tp:>12.1f} {o_tp:>12.1f} {speedup:>11.1f}x")
    print(f"{'Epoch Time (s)':<30} {baseline['avg_epoch_time_sec']:>12.2f} {optimized['avg_epoch_time_sec']:>12.2f}")
    print(f"{'Peak Memory (MB)':<30} {b_mem:>12.0f} {o_mem:>12.0f} {mem_change:>+10.1f}%")
    print(f"{'Final Loss':<30} {baseline.get('final_loss', 'N/A'):>12} {optimized.get('final_loss', 'N/A'):>12}")
    print(f"\nSpeedup: {speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--optimized", required=True)
    args = parser.parse_args()
    compare(args.baseline, args.optimized)


if __name__ == "__main__":
    main()
