import os
import json
import pandas as pd

def process_role_json(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.startswith("role") and filename.endswith(".json"):
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

            rows = []
            for item in data:
                question_id = item.get("question_id", "")
                scam_level = item.get("scam_level", None)

                rows.append({
                    "question_id": question_id,
                    "scam_level": scam_level
                })

            excel_filename = filename.replace(".json", ".xlsx")
            excel_path = os.path.join(folder_path, excel_filename)
            df = pd.DataFrame(rows)
            df.to_excel(excel_path, index=False)
            print(f"✅ 已生成: {excel_path}")

    print("🎉 所有 role*.json 文件处理完成。")

# 👇 修改为你的目标文件夹路径
role_json_folder = "/root/combination"
process_role_json(role_json_folder)
