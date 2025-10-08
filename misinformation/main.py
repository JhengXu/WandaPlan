import os
import json
from pathlib import Path
from misinfo import MisinfoFraud

from trueinfo_gene import get_trueinfo_lines, TRUEINFO_DIRS

# Environment variables (you can also export them externally)
# os.environ["MODEL_NAME"]  = "gpt-4o"
# os.environ["LLM_API_KEY"] = "sk-xxxx"
# os.environ["API_BASEURL"] = "https://api.openai.com/v1"

INPUT_REQUESTS = "synthetic_travel_requests.json"
OUT_JSON = "final_results.json"


def load_trueinfo_lines(scenario: str, qid: str, request: dict):
    p = TRUEINFO_DIRS[scenario] / f"{qid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    lines = get_trueinfo_lines(scenario, request, min_lines=3, use_cache=True, save_on_success=True)
    return lines or []


def main():
    """
    Main pipeline:
    1. Read synthetic_travel_requests.json
    2. For each request and each scenario (hotel/flight):
       - Load trueinfo lines (or auto-generate if missing)
       - Generate misinfo lines
       - Mix them and shuffle
       - Ask ranking agent to order
       - Compute evaluation metrics
    3. Save all results into final_results.json
    """
    data_raw = Path(INPUT_REQUESTS).read_text(encoding="utf-8")
    data = json.loads(data_raw)


    if isinstance(data, dict):
        reqs = []
        for k, v in data.items():
            if isinstance(v, dict):
                v["question_id"] = k
                reqs.append(v)
    else:
        reqs = data

    results = {"per_request": {}, "aggregate": {"hotel": {}, "flight": {}}}
    # Aggregators for average metrics
    agg = {
        "hotel": {"hit": {1: 0, 2: 0, 4: 0, 8: 0}, "ndcg": {1: 0, 2: 0, 4: 0, 8: 0}, "N": 0},
        "flight": {"hit": {1: 0, 2: 0, 4: 0, 8: 0}, "ndcg": {1: 0, 2: 0, 4: 0, 8: 0}, "N": 0},
    }

    for scenario in ["hotel", "flight"]:
        runner = MisinfoFraud(scenario=scenario)
        for req in reqs:
            qid = str(req.get("question_id", req.get("id", ""))).strip()
            if not qid:
                continue

            true_lines = load_trueinfo_lines(scenario, qid, req)
            if len(true_lines) < 3:
                continue

            # 1) Generate misinformation
            mis_lines = runner.misinfo_gene(true_lines)

            # 2) Mix true + fake, shuffle and re-index
            mix_lines, true_idx, mis_idx = runner.mix_info(
                trueinfo_lines=true_lines,
                misinfo_lines=mis_lines,
                ret_num=8,
                mis_ratio=0.5,
                seed=None, 
            )

            # 3) Ranking by agent
            ranking, raw = runner.rank_mix(req, mix_lines)

            # 4) Compute evaluation metrics
            metrics = runner.compute_metrics(ranking, true_idx, ks=(1, 2, 4, 8))

            # Save per-request result
            results["per_request"].setdefault(qid, {})[scenario] = {
                "true_items": true_lines,
                "mis_items": mis_lines,
                "mixed": mix_lines,
                "true_idx": true_idx,
                "mis_idx": mis_idx,
                "order": ranking,
                "agent_raw": raw,
                "metrics": metrics,
            }

            # Aggregate metrics
            agg[scenario]["N"] += 1
            for k in (1, 2, 4, 8):
                agg[scenario]["hit"][k] += metrics["hit"][k]
                agg[scenario]["ndcg"][k] += metrics["ndcg"][k]

    # Compute averages for each scenario
    for scenario in ["hotel", "flight"]:
        N = agg[scenario]["N"] or 1
        results["aggregate"][scenario] = {
            "N": N,
            "Hit@K": {k: agg[scenario]["hit"][k] / N for k in (1, 2, 4, 8)},
            "NDCG@K": {k: agg[scenario]["ndcg"][k] / N for k in (1, 2, 4, 8)},
        }

    Path(OUT_JSON).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Done. Results saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
