import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Get agent's gaia score")
    parser.add_argument(
        "--file",
        default="/Users/tangjiabin/Documents/reasoning/metachain/evaluation_results/gaia/system_triage_agent/claude-3-5-sonnet-20241022_maxiter/output.jsonl",
        type=str,
        help="Path to the agent's output.jsonl",
    )
    args = parser.parse_args()
    this_log = args.file
    outs = []
    with open(this_log, "r") as f:
        lines = f.readlines()
        for line in lines:
            outs.append(json.loads(line))
    print(f"Reading {this_log}")
    print(f'Metadata:\n {outs[0]["metadata"]}')

    total = 0
    success = 0
    l1_total = 0
    l1_success = 0
    l2_total = 0
    l2_success = 0
    l3_total = 0
    l3_success = 0
    for out in outs:
        total += 1
        if out["test_result"]["score"]:
            success += 1
        if out["instance"]["Level"] == "1":
            l1_total += 1
            if out["test_result"]["score"]:
                l1_success += 1
        elif out["instance"]["Level"] == "2":
            l2_total += 1
            if out["test_result"]["score"]:
                l2_success += 1
        elif out["instance"]["Level"] == "3":
            l3_total += 1
            if out["test_result"]["score"]:
                l3_success += 1
    print(f"Success rate: {success}/{total} = {success/total * 100:.4f}%")
    print(
        f"L1 success rate: {l1_success}/{l1_total} = {l1_success/l1_total * 100:.4f}%"
    )
    print(
        f"L2 success rate: {l2_success}/{l2_total} = {l2_success/l2_total * 100:.4f}%"
    )
    print(
        f"L3 success rate: {l3_success}/{l3_total} = {l3_success/l3_total * 100:.4f}%"
    )


if __name__ == "__main__":
    main()
