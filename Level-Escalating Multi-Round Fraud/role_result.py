import json
from collections import Counter

model_list = [
    "TA_mistralai_Mixtral-8x22B-Instruct-v0.1",
    "TA_mistralai_Mixtral-8x7B-Instruct-v0.1",
    "gpt-3.5-turbo",
    "gemini-2.5-flash-preview-04-17",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest"
    "grok-3-beta"
]

def count_scam_levels(data):
    scam_levels = [item['scam_level'] for item in data]
    return Counter(scam_levels)

def adjusted_min_level(level1, level2):
    # 将 scam_level == 0 转换为 5，仅用于比较
    val1 = 5 if level1 == 0 else level1
    val2 = 5 if level2 == 0 else level2
    min_val = min(val1, val2)
    # 如果最小值是 5，说明原始值中有 0
    return 0 if min_val == 5 else min_val

summary = {}

for model_tag in model_list:
    hotel_file = f"hotel_new_{model_tag}.json"
    flight_file = f"flight_new_{model_tag}.json"

    try:
        with open(hotel_file, 'r', encoding='utf-8') as f:
            data1 = [item for item in json.load(f) if int(item['question_id']) <= 455]
        with open(flight_file, 'r', encoding='utf-8') as f:
            data2 = [item for item in json.load(f) if int(item['question_id']) <= 455]
    except FileNotFoundError:
        print(f"[跳过] 缺少文件：{hotel_file} 或 {flight_file}")
        continue

    # 转换为字典：question_id -> item
    dict1 = {item['question_id']: item for item in data1}
    dict2 = {item['question_id']: item for item in data2}

    # 取交集
    common_ids = set(dict1.keys()) & set(dict2.keys())

    filtered_data1 = [dict1[qid] for qid in common_ids]
    filtered_data2 = [dict2[qid] for qid in common_ids]

    # 功能一：各自 scam_level 分布
    count1 = count_scam_levels(filtered_data1)
    count2 = count_scam_levels(filtered_data2)

    # 功能二：对共有 question_id，记录两个 scam_level 中“按规则较小”的那个
    merged_entries = []
    for qid in common_ids:
        level1 = dict1[qid]['scam_level']
        level2 = dict2[qid]['scam_level']
        min_level = adjusted_min_level(level1, level2)
        merged_entries.append({'question_id': qid, 'scam_level': min_level})

    merged_count = count_scam_levels(merged_entries)

    summary[model_tag] = {
        "file1_scam_levels": {str(k): count1.get(k, 0) for k in range(5)},
        "file2_scam_levels": {str(k): count2.get(k, 0) for k in range(5)},
        "min_scam_levels": {str(k): merged_count.get(k, 0) for k in range(5)}
    }

# 保存汇总结果
with open('new_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("所有模型处理完毕，结果已保存至 scam_level_summary.json")
