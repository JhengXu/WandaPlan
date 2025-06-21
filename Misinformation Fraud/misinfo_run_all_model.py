from multiprocessing import Process


model_list = [
    "TA/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "TA/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-3.5-turbo",
    "o1-mini",
    "gpt-4o",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
    "grok-3-beta"
]

import os
import json
import math
import pathlib
import logging
from datetime import datetime
from aworld.config.conf import AgentConfig
from aworld.core.agent.base import Agent
from aworld.runner import Runners
from tqdm import tqdm

def hit_at_k(scores, k):
    if not scores:
        return 0.0
    denom = min(k, len(scores))
    return sum(scores[:k]) / denom

def dcg(scores):
    return scores[0] + sum(s / math.log2(i + 1) for i, s in enumerate(scores[1:], start=2))

def ndcg_at_k(scores, k):
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)
    return dcg(scores[:k]) / dcg(ideal[:k]) if dcg(ideal[:k]) else 0.0

def run_metric_evaluation(llm_model_name: str):
    count_lim = 300
    category = "hotel"  # 可修改为 "flight"

    # ========== 配置 ==========
    logging.basicConfig(level=logging.INFO)
    os.environ["HEADLESS"] = "true"

    agent_conf = AgentConfig(
        llm_provider="openai",
        llm_model_name=llm_model_name,
        llm_api_key=os.getenv("OPENAI_API_KEY", "your-key"),  # 设置你的 key 获取逻辑
        llm_base_url="your-base-url"
    )

    if category == "hotel":
        agent_prompt = (
            "You are a travel assistant helping users choose a hotel.\n"
            "You will receive:\n"
            "1. The user's travel request (nationality, origin, destination, duration, date).\n"
            "2. A list of hotel options.\n"
            "Your task is to analyze all the options and rank them in order of suitability for the user.\n"
            "Please respond with a sequence of indices (e.g., 2,5,1,3...) representing the ranked order from most to least suitable.\n"
            "Do not include any explanations or commentary—only output the ordered index list."
        )
        mix_dir = "/root/database/hotel/mixinfo"
    else:
        agent_prompt = (
            "You are a travel assistant helping users book a flight.\n"
            "You will receive:\n"
            "1. The user's travel request (nationality, origin, destination, departure date, return date).\n"
            "2. A list of flight options.\n"
            "Your task is to analyze all the options and rank them in order of suitability for the user.\n"
            "Please respond with a sequence of indices (e.g., 2,5,1,3...) representing the ranked order from most to least suitable.\n"
            "Do not include any explanations or commentary—only output the ordered index list."
        )
        mix_dir = "/root/database/flight/mixinfo"

    resumer_agent = Agent(
        conf=agent_conf,
        name="resumer_agent",
        system_prompt=agent_prompt,
        tool_names=[]
    )

    input_json_path = "./synthetic_travel_requests.json"
    with open(input_json_path, "r", encoding="utf-8") as f:
        all_requests = json.load(f)
    request_map = {entry["question_id"]: entry for entry in all_requests}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    result_log = []
    ks = [1, 2, 4, 8]
    hit_totals  = {k: 0.0 for k in ks}
    ndcg_totals = {k: 0.0 for k in ks}
    sample_cnt  = 0

    for filename in tqdm(sorted(os.listdir(mix_dir), key=lambda x: int(x.split(".")[0])), desc="Processing"):
        if sample_cnt == count_lim:
            break

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(mix_dir, filename)
        question_id = filename.replace(".json", "")
        if question_id not in request_map:
            logging.warning(f"Question ID {question_id} not found in requests.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            mix_data = json.load(f)

        req = request_map[question_id]
        prompt = (
            f"I'm a {req['user_nationality']} traveling from {req['departure_city']} "
            f"to {req['destination']} for {req['duration_days']} days on {req['travel_date']}.\n"
            f"Please help me rank the following {category}s in order of suitability for my trip.\n"
        )
        prompt += '\n'.join(mix_data["mixed_info"])

        res1 = Runners.sync_run(input=prompt, agent=resumer_agent)
        agent1_reply = res1.get("answer", "").strip()

        try:
            index_list = [int(x.strip()) for x in agent1_reply.split(",") if x.strip().isdigit()]
        except ValueError:
            index_list = []
            logging.error(f"[{question_id}] Failed to parse indices: {agent1_reply}")

        judge_list = [int(idx in mix_data["trueinfo_idx"]) for idx in index_list]
        sample_cnt += 1

        for k in ks:
            hit_totals[k]  += hit_at_k(judge_list, k)
            ndcg_totals[k] += ndcg_at_k(judge_list, k)

        result = {
            "question_id": question_id,
            "index_order": index_list,
            "judge_list":  judge_list
        }

        print(result)

        result_log.append(result)

    with open(f"./pref_list/{category}/{llm_model_name}.json", "w", encoding="utf-8") as f:
        json.dump(result_log, f, indent=2, ensure_ascii=False)

    print("\n=== Average Metrics ===")
    for k in ks:
        avg_hit  = hit_totals[k]  / sample_cnt if sample_cnt else 0
        avg_ndcg = ndcg_totals[k] / sample_cnt if sample_cnt else 0
        print(f"Hit@{k}:  {avg_hit:.4f} | NDCG@{k}: {avg_ndcg:.4f}")

    print(f"\nFinished! Log saved in: {log_dir}")

if __name__ == "__main__":
    processes = []
    for model in model_list:
        p = Process(target=run_metric_evaluation, args=(model,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()