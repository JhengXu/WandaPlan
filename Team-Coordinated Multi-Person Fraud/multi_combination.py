import os
import json
import pandas as pd

def process_multi_json_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.startswith("multi") and filename.endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            print(f"📖 正在读取: {json_path}")

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    print(f"⚠️ 跳过 {filename}：不是列表结构")
                    continue
            except Exception as e:
                print(f"❌ 无法解析 {filename}: {e}")
                continue

            # 组织成 question_id 行，n_scam_agents 列
            result = {}
            all_agent_counts = set()

            for item in data:
                qid = item.get("question_id")
                agent_count = item.get("n_scam_agents")
                scammed = item.get("scammed")

                all_agent_counts.add(agent_count)
                if qid not in result:
                    result[qid] = {}
                result[qid][agent_count] = scammed

            # 排序列名
            sorted_agent_counts = sorted(all_agent_counts)
            rows = []

            for qid, scams in result.items():
                row = {"question_id": qid}
                for count in sorted_agent_counts:
                    row[count] = scams.get(count, "")
                rows.append(row)

            # 保存为 Excel
            df = pd.DataFrame(rows)
            df = df[["question_id"] + sorted_agent_counts]  # 保证列顺序
            excel_filename = filename.replace(".json", ".xlsx")
            excel_path = os.path.join(folder_path, excel_filename)
            df.to_excel(excel_path, index=False)
            print(f"✅ 已生成: {excel_path}")

    print("🎉 所有 multi*.json 文件处理完毕。")

# 替换为你的实际路径
multi_json_folder = "/root/combination"
process_multi_json_files(multi_json_folder)
