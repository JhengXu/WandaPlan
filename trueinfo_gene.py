# trueinfo_gene.py — unified true-info generator for A/B/C pipelines
# -*- coding: utf-8 -*-
"""
Public API
----------
get_trueinfo_lines(scenario, request, *, min_lines=3, use_cache=True, save_on_success=True, ...)

- scenario: "hotel" | "flight"
- request: {
    "question_id": str,
    "user_nationality": str,
    "departure_city": str,
    "destination": str,
    "duration_days": int,
    "travel_date": str,
  }

Behavior
--------
1) If cache (./trueinfo|./flight/{qid}.json) exists and use_cache=True, return cached lines.
2) Otherwise call crawler+extractor (browser_async) under allowed domains to extract visible listings.
3) If >= min_lines, save to cache (when save_on_success=True) and return; else return [].
"""

from pathlib import Path
import os
import json
import logging
from typing import Dict, List

# ================ Shared cache directories ================
TRUEINFO_DIRS: Dict[str, Path] = {
    "hotel": Path("./hotel"),
    "flight": Path("./flight"),
}

# Ensure headless browsing by default
os.environ.setdefault("HEADLESS", "true")

# ==================== Config via env ====================
DEFAULT_MODEL = os.getenv("MODEL_NAME") or os.getenv("LLM_MODEL_NAME") or "gpt-4o"
DEFAULT_PROVIDER = os.getenv("PROVIDER") or "openai"
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "your-key")
DEFAULT_BASE_URL = os.getenv("API_BASEURL", "your-base-url")

# ==================== Site allowlist ====================
CATEGORY_SITES: Dict[str, List[str]] = {
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

# ==================== Lazy agent singletons ====================
_CRAWLER_AGENTS: Dict[str, "Agent"] = {}
_EXTRACTOR_AGENTS: Dict[str, "Agent"] = {}

def _get_agents(category: str):
    """
    Lazily create and reuse crawler/extractor agents.
    Heavy project modules are imported inside this function to avoid
    import errors in environments that don't have them.
    """
    from aworld.config.conf import AgentConfig
    from aworld.core.agent.base import Agent

    def _create_crawler_agent(category: str) -> "Agent":
        domains = CATEGORY_SITES[category]
        domain_list = "\n".join(f"- {url}" for url in domains)
        agent_conf = AgentConfig(
            llm_provider=DEFAULT_PROVIDER,
            llm_model_name=DEFAULT_MODEL,
            llm_api_key=DEFAULT_API_KEY,
            llm_base_url=DEFAULT_BASE_URL
        )
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

    def _create_extractor_agent(category: str) -> "Agent":
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

        agent_conf = AgentConfig(
            llm_provider=DEFAULT_PROVIDER,
            llm_model_name=DEFAULT_MODEL,
            llm_api_key=DEFAULT_API_KEY,
            llm_base_url=DEFAULT_BASE_URL
        )
        return Agent(
            conf=agent_conf,
            name=f"{category}_extractor_agent",
            system_prompt=detail_prompt,
            tool_names=["browser_async"]
        )

    if category not in _CRAWLER_AGENTS:
        _CRAWLER_AGENTS[category] = _create_crawler_agent(category)
    if category not in _EXTRACTOR_AGENTS:
        _EXTRACTOR_AGENTS[category] = _create_extractor_agent(category)
    return _CRAWLER_AGENTS[category], _EXTRACTOR_AGENTS[category]

# ==================== Retry heuristic ====================
def _needs_retry_extraction(output: str, min_lines: int) -> bool:
    low = (output or "").lower()
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    refusal = any(phrase in low for phrase in [
        "unable to", "can't access", "cannot access", "cannot assist",
        "can't assist", "provide me with", "as an ai", "i do not have access"
    ])
    return refusal or (len(lines) < min_lines)

# ==================== Core crawl+extract ====================
def _fetch_visible_lines_for(category: str, user_prompt: str, *, min_lines: int, max_retries: int) -> List[str]:
    from aworld.runner import Runners
    crawler, extractor = _get_agents(category)

    # Stage 1: Crawler
    r1 = Runners.sync_run(input=user_prompt, agent=crawler)
    urls_text = r1.get("answer", "").strip()

    # Stage 2: Extractor with retry
    retry = 0
    output = ""
    while True:
        r2 = Runners.sync_run(input=urls_text, agent=extractor)
        output = r2.get("answer", "").strip()
        if not _needs_retry_extraction(output, min_lines=min_lines):
            break
        retry += 1
        if retry >= max_retries:
            break
        logging.warning(f"[{category}] extractor retry {retry}/{max_retries} due to weak output.")

    lines = [ln for ln in output.splitlines() if ln.strip()]
    return lines

# ==================== Public API ====================
def get_trueinfo_lines(
    scenario: str,
    request: Dict,
    *,
    min_lines: int = 3,
    use_cache: bool = True,
    save_on_success: bool = True,
    out_dirs: Dict[str, Path] = TRUEINFO_DIRS,
    max_retries: int = 5,
) -> List[str]:
    """
    Unified entry point for A/B/C groups.
    Returns a list of visible listings. [] if extraction fails or < min_lines.
    """
    if scenario not in out_dirs:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected keys in out_dirs: {list(out_dirs.keys())}")

    qid = request.get("question_id")
    if not qid:
        raise ValueError("request must contain 'question_id'.")

    out_dir = out_dirs[scenario]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{qid}.json"

    # 1) read cache
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception as e:
            logging.warning(f"Failed to read cache {cache_path}: {e}. Will refetch.")

    # 2) build user prompt
    prompt_template = "I'm a XX traveling from YY to ZZ for AA days on BB."
    user_prompt = (
        prompt_template
        .replace("XX", str(request.get("user_nationality", "")))
        .replace("YY", str(request.get("departure_city", "")))
        .replace("ZZ", str(request.get("destination", "")))
        .replace("AA", str(request.get("duration_days", "")))
        .replace("BB", str(request.get("travel_date", "")))
    ).strip()

    # 3) fetch
    try:
        lines = _fetch_visible_lines_for(
            scenario,
            user_prompt,
            min_lines=min_lines,
            max_retries=max_retries
        )
    except Exception as e:
        logging.error(f"[{scenario}] fetching trueinfo failed for qid={qid}: {e}")
        return []

    # 4) save if valid
    if len(lines) >= min_lines:
        if save_on_success:
            try:
                cache_path.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as e:
                logging.warning(f"Failed to write cache {cache_path}: {e}")
        return lines

    # Not enough lines
    logging.info(f"[{scenario}] qid={qid} only {len(lines)} items (<{min_lines}); returning [].")
    return []

# ==================== Convenience helpers ====================
def load_cached_trueinfo(scenario: str, qid: str, out_dirs: Dict[str, Path] = TRUEINFO_DIRS) -> List[str]:
    """Load cached trueinfo lines if present; otherwise []."""
    p = out_dirs[scenario] / f"{qid}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            logging.warning(f"Failed to read cache {p}: {e}")
    return []

def save_trueinfo(scenario: str, qid: str, lines: List[str], out_dirs: Dict[str, Path] = TRUEINFO_DIRS) -> None:
    """Force save trueinfo lines (no validation)."""
    out_dirs[scenario].mkdir(parents=True, exist_ok=True)
    p = out_dirs[scenario] / f"{qid}.json"
    p.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding='utf-8')
