import os
from round import RoundFraud

API_KEY   = os.getenv("OPENAI_API_KEY", "")
BASE_URL  = os.getenv("API_BASEURL", "https://api.openai.com/v1")
PROVIDER  = os.getenv("PROVIDER", "openai")

MODEL_UNDER_TEST = "gpt-4o-mini"
TRAVEL_REQUESTS  = "./synthetic_travel_requests.json"
CATEGORIES       = ("flight", "hotel")
ROUNDS           = 4

# Optional role models
CRAWL_MODEL   = "gpt-4o"
EXTRACT_MODEL = "gpt-4o"
SCAMMER_MODEL = "gpt-4o"

OUT_JSON   = "all_results.json"
TRUE_DIR   = "./trueinfo"
LOGS_DIR   = "./logs"

def main():
    pipeline = RoundFraud(
        api_key=API_KEY,
        base_url=BASE_URL,
        provider=PROVIDER,
        judge_model="gpt-4o",  # judge used to return YES/NO
        headless=True,
    )
    pipeline.run_all_and_save(
        model_under_test=MODEL_UNDER_TEST,
        travel_requests=TRAVEL_REQUESTS,
        out_json_path=OUT_JSON,
        categories=CATEGORIES,
        crawl_model=CRAWL_MODEL,
        extract_model=EXTRACT_MODEL,
        scammer_model=SCAMMER_MODEL,
        rounds=ROUNDS,
        write_trueinfo=True,   # save crawled true info
        trueinfo_dir=TRUE_DIR,
        write_logs=True,       # save per-question logs
        logs_dir=LOGS_DIR,
    )

if __name__ == "__main__":
    main()
