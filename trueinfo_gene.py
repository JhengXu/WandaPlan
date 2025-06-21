import os
import pathlib
import logging
import json
from datetime import datetime
from tqdm import tqdm
from aworld.config.conf import AgentConfig
from aworld.core.agent.base import Agent
from aworld.runner import Runners

trueinfo_dir = "/root/database/hotel/trueinfo"

# ==================== 配置 ====================
os.environ["HEADLESS"] = "true"

agent_conf = AgentConfig(
    llm_provider="openai",
    llm_model_name="gpt-4o",
    llm_api_key=os.getenv("OPENAI_API_KEY", "your-key"),
    llm_base_url="your-base-url"
)

# ==================== 工具和网站映射 ====================
CATEGORY_SITES = {
    "flight": [
        "https://www.skyscanner.net",
        "https://www.google.com/flights",
        "https://flights.ctrip.com",
        "https://www.fliggy.com"
    ],
    "hotel": [
        "https://www.booking.com",
        "https://www.agoda.com",
        "https://www.trip.com",
        "https://www.expedia.com",
        "https://www.hotels.com",
        "https://www.airbnb.com",
        "https://www.trivago.com"
    ],
}

# ==================== Agent 创建器 ====================
def create_crawler_agent(category):
    domains = CATEGORY_SITES[category]
    domain_list = "\n".join(f"- {url}" for url in domains)
    return Agent(
        conf=agent_conf,
        name=f"{category}_crawler_agent",
        system_prompt=(
            f"You are a crawler agent specializing in {category}. "
            f"Your role is to locate real and relevant web pages for a user's travel plan.\n\n"
            f"Only search within the following allowed domains:\n{domain_list}\n\n"
            f"Follow these steps for each domain:\n"
            f"1. Use the browser tool to search for pages related to the user's request (e.g., listings, offers, guides).\n"
            f"2. Navigate into the search results or listings. Allow each page to fully load.\n"
            f"3. Execute JavaScript (e.g., `window.location.href`) to get the final URL after navigation.\n"
            f"4. For each site, return at least **3 unique final URLs** that directly relate to {category}.\n\n"
            f"Do not include any commentary, explanation, or irrelevant text.\n"
            f"Your output should be a plain list of final URLs, one per line, and nothing else."
        ),
        tool_names=["browser_async"]
    )

def create_extractor_agent(category):
    if category == "flight":
        detail_prompt = (
            f"You are a professional flight information extractor. "
            f"You will be given a list of URLs, each pointing to a page containing relevant flight-related information.\n\n"
            f"For each URL:\n"
            f"1. Use the browser_async tool to load the page.\n"
            f"2. Read and interpret the on-page content carefully.\n"
            f"3. Extract multiple **real and distinct** listings based solely on visible information.\n"
            f"4. Avoid duplication across listings.\n\n"
            f"You are supposed to follow the structured format:\n\n"
            f"   1. **<Platform> (Route)**\n"
            f"      - **Airline:** <Airline name>\n"
            f"      - **Route:** <From – To, with info like Direct or 1 stop>\n"
            f"      - **Price:** <Round-trip or one-way price with currency>\n"
            f"      - **Departure/Arrival:** <Departure and arrival time, if available>\n"
            f"      - **Booking Method:** <Official site, aggregator, OTA, etc.>\n"
            f"Use Arabic numerals to number each entry: 1., 2., 3., ...\n\n"
            f"Do NOT guess, fabricate, or include generic assumptions.\n"
            f"Output should be a clean list, with one valid fact or listing per line, without URL.\n"
            f"Keep it concise, fact-based, and specific to what appears on the actual page."
        )
    else:  # hotel
        detail_prompt = (
            f"You are a professional hotel information extractor. "
            f"You will be given a list of URLs, each pointing to a page containing relevant hotel-related information.\n\n"
            f"For each URL:\n"
            f"1. Use the browser_async tool to load the page.\n"
            f"2. Read and interpret the on-page content carefully.\n"
            f"3. Extract multiple **real and distinct** listings based solely on visible information.\n"
            f"4. Avoid duplication across listings.\n\n"
            f"You are supposed to follow the structured format:\n\n"
            f"   1. **<Platform> (City, Country)**\n"
            f"      - **Hotel Name:** <Hotel name>\n"
            f"      - **Location:** <Brief address or landmark>\n"
            f"      - **Price:** <Nightly price with currency>\n"
            f"      - **Rating:** <Rating and optionally review count>\n"
            f"      - **Features:** <Key features, amenities, or services>\n"
            f"Use Arabic numerals to number each entry: 1., 2., 3., ...\n\n"
            f"Do NOT guess, fabricate, or include generic assumptions.\n"
            f"Output should be a clean list, with one valid fact or listing per line, without URL.\n"
            f"Keep it concise, fact-based, and specific to what appears on the actual page."
        )


    return Agent(
        conf=agent_conf,
        name=f"{category}_extractor_agent",
        system_prompt=detail_prompt,
        tool_names=["browser_async"]
    )

# ==================== Agent 初始化 ====================
categories = ["flight"]
crawler_agents = {cat: create_crawler_agent(cat) for cat in categories}
extractor_agents = {cat: create_extractor_agent(cat) for cat in categories}

# ==================== 辅助函数：检测 extractor 是否调用了浏览器工具 ====================
def needs_retry_extraction(output: str) -> bool:
    low = output.lower()
    lines = [line for line in output.splitlines() if line.strip()]
    need = any(phrase in low for phrase in [
        "unable to", "can't access", "if you provide me",
        "cannot assist", "can't assist"
    ]) or len(lines) > 3
    return 


# ==================== 主流程 ====================
def main():
    # 确保输出目录存在
    for dir_name in categories:
        pathlib.Path(dir_name).mkdir(exist_ok=True)

    # 读取旅行请求 JSON
    with open("./synthetic_travel_requests.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    entry_count = 0
    for entry in tqdm(data, desc="Processing travel requests"):
        # 处理 entry
        question_id = entry["question_id"]
        user_nationality = entry["user_nationality"]
        departure_city = entry["departure_city"]
        destination = entry["destination"]
        duration_days = entry["duration_days"]
        travel_date = entry["travel_date"]

        if os.path.exists(f"{trueinfo_dir}/{question_id}.json"):
            continue

        # 构造用户 prompt
        prompt_template = "I'm a XX traveling from YY to ZZ for AA days on BB."
        user_prompt = (
            prompt_template
            .replace("XX", user_nationality)
            .replace("YY", departure_city)
            .replace("ZZ", destination)
            .replace("AA", str(duration_days))
            .replace("BB", travel_date)
        )

        # 处理每个分类
        entry_count += 1
        for cat in categories:
            # Stage 1: Crawler
            print(f"[{cat.upper()}] Crawling URLs for question_id={question_id}...")
            r1 = Runners.sync_run(input=user_prompt, agent=crawler_agents[cat])
            urls = r1.get("answer", "").strip()

            # Stage 2: Extractor (含重试)
            print(f"[{cat.upper()}] Extracting real info for question_id={question_id}...")
            r2 = Runners.sync_run(input=urls, agent=extractor_agents[cat])
            true_info = r2.get("answer", "").strip()

            retry_count = 0
            while(needs_retry_extraction(true_info) and retry_count < 5):
                logging.warning(f"{cat}_extractor_agent did not use browser. Retrying once...")
                r2 = Runners.sync_run(input=urls, agent=extractor_agents[cat])
                true_info = r2.get("answer", "").strip()
                retry_count += 1

            # 保存到对应的数据库文件夹，仅当条目 >= 3 行时保存 JSON
            lines = [line for line in true_info.splitlines() if line.strip()]
            if len(lines) >= 3:
                out_folder = cat  # 'flight' 或 'hotel'
                json_path = os.path.join(out_folder, f"{question_id}.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(lines, jf, ensure_ascii=False, indent=2)
            else:
                logging.info(f"Skipping {cat} for question_id={question_id}: only {len(lines)} items found (<3)")

    print("All entries processed. Database files saved in 'hotel/' and 'flight/' folders (where applicable).")

if __name__ == "__main__":
    main()