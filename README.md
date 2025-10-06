# 📘 WandaPlan: Fraud Evaluation for LLM Travel Planners

**WandaPlan** is a plug-and-play evaluation suite for testing how LLM agents handle **fraudulent content** in real-world **travel planning**. It provides three modular scenarios you can drop into any agent pipeline:

* **Misinformation Fraud** — inject deceptive listings and test ranking robustness
* **Team-Coordinated Multi-Person Fraud** — simulate a group of scammers persuading a user to switch
* **Level-Escalating Multi-Round Fraud** — simulate a “customer-service” scam that escalates over several rounds

These scenarios map to the paper *“Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? Exploring Fraud Detection in Travel Planning.”*

---

## 📂 Repository Layout

```
.
├── synthetic_travel_requests.json     # Test queries (user nationality, route, date, etc.)
├── trueinfo_gene.py                   # Shared module for generating/loading true-info
├── misinformation/
│   ├── main.py                        # CLI: run misinfo pipeline
│   ├── misinfo.py                     # MisinfoFraud class
│   └── prompts.py                     # Prompts for misinfo
├── multi-person/
│   ├── main.py                        # CLI: run team-coordinated (N scammers)
│   ├── team.py                        # TeamFraud class
│   └── prompt.py                      # Prompts for team fraud
└── multi-round/
    ├── main.py                        # CLI: run multi-round escalating scam
    ├── round.py                       # RoundFraud class
    └── prompt.py                      # Prompts for crawler/extractor/user/scammer/judge
```

> Each submodule is **standalone** and can be used independently.
> All three read from the shared `synthetic_travel_requests.json`.

---

## ⚙️ Requirements

* **Python 3.9+**
* An LLM runtime (default assumes **OpenAI-compatible API**)
* Packages:

```bash
pip install -U tqdm pandas openpyxl
# plus your agent framework deps, e.g.:
# pip install aworld
```

---

## 🔑 Environment Variables

All groups (A/B/C) now use the same environment variables:

```bash
export OPENAI_API_KEY="sk-..."           # main API key
export API_BASEURL="https://api.openai.com/v1"   # or your custom gateway
export PROVIDER="openai"                 # e.g., "openai"
export MODEL_NAME="gpt-4o-mini"          # default test model
```

👉 **Compatibility:**

* `LLM_API_KEY` is still accepted as a fallback to `OPENAI_API_KEY` (for older configs).
* `HEADLESS=true` is auto-set in the code for browser/tool agents.

---

## 📄 Data Format

Your requests JSON should be a list of dicts:

```json
[
  {
    "question_id": 1,
    "user_nationality": "Singaporean",
    "departure_city": "Helsinki",
    "destination": "Warsaw",
    "duration_days": 7,
    "travel_date": "2025-08-18"
  }
]
```

---

## 🚀 How to Run

> All commands run from the repo root.

### A) Misinformation Fraud

```bash
python misinformation/main.py \
  --api_key "$OPENAI_API_KEY" \
  --base_url "$API_BASEURL" \
  --model "$MODEL_NAME" \
  --requests ./synthetic_travel_requests.json \
  --out ./outputs/misinformation_results.json
```

* Loads or generates **true info** (via `trueinfo_gene.py`)
* Fabricates **misinfo** listings
* Mixes true+fake → asks model to **rank**
* Computes **P@K** and **NDCG@K**
* Outputs `./outputs/misinformation_results.json`

---

### B) Team-Coordinated Multi-Person Fraud

```bash
python multi-person/main.py \
  --api_key "$OPENAI_API_KEY" \
  --base_url "$API_BASEURL" \
  --model "$MODEL_NAME" \
  --requests ./synthetic_travel_requests.json \
  --agents 4 \
  --out ./outputs/multiperson_results.json
```

* User selects one **true** option
* N scam agents sequentially persuade
* User responds; judge auto-labels “scammed / not scammed”
* Aggregates per-N and per-scenario

**Outputs:** `multiperson_results.json` + logs (if enabled)

---

### C) Level-Escalating Multi-Round Fraud

```bash
python multi-round/main.py \
  --api_key "$OPENAI_API_KEY" \
  --base_url "$API_BASEURL" \
  --provider "$PROVIDER" \
  --model "$MODEL_NAME" \
  --requests ./synthetic_travel_requests.json \
  --rounds 4 \
  --trueinfo_dir ./trueinfo \
  --logs_dir ./logs \
  --out ./outputs/multiround_all_results.json
```

* Crawl & extract **true info** from whitelisted domains
* Run **4 scam rounds** (Base → Credibility → Urgency → Emotional)
* Judge returns YES/NO each round; first YES sets scam level
* Saves consolidated JSON

**Outputs:**

* `multiround_all_results.json`
* `./trueinfo/` (optional true info cache)
* `./logs/` (optional full conversations)

---

## 📊 Metrics

* **Defense Success Rate (DSR)** — % of requests where user kept authentic option
* **P@K** — Precision@K on misinfo ranking
* **NDCG@K** — Order-sensitive ranking quality

---

## 🛠️ Tips & Notes

* **Rate limits**: add retries/backoff if needed
* **Determinism**: use `temperature=0` and set seeds for sampling
* **Custom providers**: change `PROVIDER`/`BASE_URL` to HuggingFace, Azure, etc.
* **trueinfo_gene.py**: can be reused by any module if you want to pre-build true-info database.

---

## ✅ One-Command Examples

```bash
# Misinfo
python misinformation/main.py --model "$MODEL_NAME" --requests ./synthetic_travel_requests.json --out ./outputs/misinformation_results.json

# Multi-Person (N=4)
python multi-person/main.py --model "$MODEL_NAME" --requests ./synthetic_travel_requests.json --agents 4 --out ./outputs/multiperson_results.json

# Multi-Round (R=4 + true-info caching + logs)
python multi-round/main.py --model "$MODEL_NAME" --requests ./synthetic_travel_requests.json --rounds 4 --trueinfo_dir ./trueinfo --logs_dir ./logs --out ./outputs/multiround_all_results.json
