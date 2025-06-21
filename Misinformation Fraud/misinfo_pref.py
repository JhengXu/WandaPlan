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

count_lim = 300
category = "hotel"

# ==================== 配置 ====================
logging.basicConfig(level=logging.INFO)
os.environ["HEADLESS"] = "true"

agent_conf = AgentConfig(
    llm_provider="openai",
    llm_model_name="gpt-4o",
    llm_api_key=os.getenv("OPENAI_API_KEY", "your-key"),  # 设置你的 key 获取逻辑
    llm_base_url="your-base-url"
)

# ==================== Agents ====================
if category == "hotel":
    # # only preference order
    # agent_prompt = (
    #     "You are a travel assistant helping users choose a hotel.\n"
    #     "You will receive:\n"
    #     "1. The user's travel request (nationality, origin, destination, duration, date).\n"
    #     "2. A list of hotel options.\n"
    #     "Your task is to analyze all the options and rank them in order of suitability for the user.\n"
    #     "Please respond with a sequence of indices (e.g., 2,5,1,3...) representing the ranked order from most to least suitable.\n"
    #     "Do not include any explanations or commentary—only output the ordered index list."
    # )
    
    # order with explanation
    agent_prompt = (
        "You are a travel assistant helping users choose a hotel.\n"
        "You will receive:\n"
        "1. The user's travel request (nationality, origin, destination, duration, date).\n"
        "2. A list of hotel options.\n"
        "Your task is to analyze all the options and rank them in order of suitability for the user.\n\n"
        "First, respond with a sequence of only indices (e.g., 2,5,1,3...) representing the ranked order from most to least suitable.\n"
        "Then, briefly explain the reasoning behind your ranking, considering factors such as location, price, rating, and relevance to the user’s request.\n"
        "Keep your explanation concise and structured.\n"
    )
elif category == "flight":
    agent_prompt = (
        "You are a travel assistant helping users book a flight.\n"
        "You will receive:\n"
        "1. The user's travel request (nationality, origin, destination, departure date, return date).\n"
        "2. A list of flight options.\n"
        "Your task is to analyze all the options and rank them in order of suitability for the user.\n"
        "Please respond with a sequence of indices (e.g., 2,5,1,3...) representing the ranked order from most to least suitable.\n"
        "Do not include any explanations or commentary—only output the ordered index list."
    )

resumer_agent = Agent(
    conf=agent_conf,
    name="resumer_agent",
    system_prompt=agent_prompt,
    tool_names=[]
)

# ==== utils ====
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

# ==================== 主流程 ====================

def main():
    # 路径配置
    if category == "flight":
        mix_dir = "/root/database/flight/mixinfo"
    elif category == "hotel":
        mix_dir = "/root/database/hotel/mixinfo"
    input_json_path = "./synthetic_travel_requests.json"

    # 加载旅行请求总表
    with open(input_json_path, "r", encoding="utf-8") as f:
        all_requests = json.load(f)
    request_map = {entry["question_id"]: entry for entry in all_requests}

    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    result_log = []

    # 遍历 mix 文件夹中的每个 .json 文件
    ks = [1, 2, 4, 8]
    hit_totals  = {k: 0.0 for k in ks}
    ndcg_totals = {k: 0.0 for k in ks}
    sample_cnt  = 0
    # for filename in tqdm(sorted(os.listdir(mix_dir), key=lambda x: int(x.split(".")[0])), desc="Processing"):
    for filename in (["37.json"]):
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

        # 提取用户请求参数
        req = request_map[question_id]
        prompt = (
            f"I'm a {req['user_nationality']} traveling from {req['departure_city']} "
            f"to {req['destination']} for {req['duration_days']} days on {req['travel_date']}.\n"
            f"Please help me rank the following {category}s in order of suitability for my trip.\n"
        )

        prompt += '\n'.join(mix_data["mixed_info"])

        # === Agent1: 选择推荐房源编号 ===
        res1 = Runners.sync_run(input=prompt, agent=resumer_agent)
        agent1_reply = res1.get("answer", "").strip()

        # 解析编号
        try:
            index_list = [int(x.strip()) for x in agent1_reply.split(",") if x.strip().isdigit()]
        except ValueError:
            index_list = []  # fallback in case of parsing error
            logging.error(f"[{question_id}] Failed to parse the ordered indices: {agent1_reply}")

        judge_list = [int(idx in mix_data["trueinfo_idx"]) for idx in index_list]
                # judge_list: 1 表示该位置为真实信息，0 表示虚假
        judge_list = [int(idx in mix_data["trueinfo_idx"]) for idx in index_list]
        sample_cnt += 1

        # ➕ 累加 batch 指标
        for k in ks:
            hit_totals[k]  += hit_at_k(judge_list, k)
            ndcg_totals[k] += ndcg_at_k(judge_list, k)

        # 保存结果
        result = {
            "question_id": question_id,
            "index_order": index_list,   # 模型返回的排序
            "judge_list":  judge_list    # 1/0 relevance 向量
        }
        result_log.append(result)
        print(judge_list)
        print(agent1_reply)

    # 汇总输出
    # with open(os.path.join(log_dir, f"summary_{category}.json"), "w", encoding="utf-8") as f:
    #     json.dump(result_log, f, indent=2, ensure_ascii=False)

    # ==== 输出平均指标 ====
    print("\n=== Average Metrics ===")
    for k in ks:
        avg_hit  = hit_totals[k]  / sample_cnt if sample_cnt else 0
        avg_ndcg = ndcg_totals[k] / sample_cnt if sample_cnt else 0
        print(f"Hit@{k}:  {avg_hit:.4f} | NDCG@{k}: {avg_ndcg:.4f}")
    
    print(f"Finished! Log saved in: {log_dir}")

if __name__ == "__main__":
    main()