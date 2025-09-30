# -*- coding: utf-8 -*-
"""
How to run:

export OPENAI_API_KEY="sk-..."
# If you use non-default base URL or provider:
# export API_BASEURL="https://api.openai.com/v1"
# export PROVIDER="openai"

python3 main.py
"""

import os
from team import TeamFraud

# Required
API_KEY   = os.getenv("OPENAI_API_KEY", "")
BASE_URL  = os.getenv("API_BASEURL", "https://api.openai.com/v1")
PROVIDER  = os.getenv("PROVIDER", "openai")

MODEL_UNDER_TEST = "gpt-4o-mini"  # model to be judged by the team scam flow
TRAVEL_REQUESTS  = "./synthetic_travel_requests.json"
CATEGORIES       = ("flight", "hotel")       # or ("flight",) / ("hotel",)
TEAM_SIZES       = (1, 2, 3, 4)              # scammers per scene
OUT_JSON         = "all_results.json"

# Optional: you can split “roles models”
CRAWL_MODEL   = "gpt-4o"
EXTRACT_MODEL = "gpt-4o"
SCAMMER_MODEL = "gpt-4o"

def main():
    tf = TeamFraud(
        api_key=API_KEY,
        base_url=BASE_URL,
        provider=PROVIDER,
        headless=True,
    )
    tf.run_all_and_save(
        model_name=MODEL_UNDER_TEST,
        travel_requests=TRAVEL_REQUESTS,
        out_json_path=OUT_JSON,
        categories=CATEGORIES,
        n_scam_agents_list=TEAM_SIZES,
        crawl_model=CRAWL_MODEL,
        extract_model=EXTRACT_MODEL,
        scammer_model=SCAMMER_MODEL,
    )

if __name__ == "__main__":
    main()
