

<div align="center">

## Is Your LLM-Based Multi-Agent a Reliable Real-World Planner? <br> Exploring Fraud Detection in Travel Planning


Junchi Yao*, Jianhua Xu*, Tianyu Xin*, Ziyi Wang, Shenzhe Zhu, Shu Yang†, Di Wang†

(*Contribute equally, †Corresponding author)

[**📝 arxiv**](https://doi.org/10.48550/arXiv.2505.16557)

</div>

## 📰 News
- **2025/06/21**: ❗️We have released our code.
- **2025/06/13**:  😍 Our paper is accepted by ICML 2025 Workshop MAS

## Introduction

WandaPlan is an evaluation environment designed to assess the fraud detection capabilities of Large Language Model (LLM)-based multi-agent planning systems in real-world scenarios such as travel planning. By incorporating real-world data with deceptive content, WandaPlan simulates realistic fraud scenarios to comprehensively evaluate the vulnerability and reliability of planning systems.

## Main Contributions

1. **WandaPlan Environment**: A novel evaluation environment based on real-world data, injected with fraudulent information and scammers to assess the risk of real-world open-source planning frameworks.
2. **Identification of Vulnerabilities**: Reveals significant weaknesses in existing frameworks that prioritize task efficiency over data authenticity, addressing a critical research gap.
3. **Mitigation Strategies**: Proposes integrating an anti-fraud agent into the travel planning framework to enhance resilience against online fraud, significantly improving reliability.

## Data Files

- **synthetic_travel_requests.json**: Contains 1000 synthetic travel requests, each including user nationality, departure city, destination, travel duration, and travel date.

## Code Files

### Real Information Collection and Processing

- **trueinfo_gene.py**: Collects real flight and hotel information from real-world data sources and generates corresponding JSON files.
- **misinfo_hotel.py** and **misinfo_flight.py**: Generate files that mix real and fraudulent hotel/flight information.
- **misinfo_run_all_model.py** and **misinfo_safe.py**: Evaluate multiple models and include metrics to assess ranking quality and fraud resistance.

### Fraud Scenario Simulation

- **role_flight.py** and **role_hotel.py**: Simulate interactions between users and fraud agents in travel planning to evaluate decision-making under fraudulent information.
- **role_combination.py**: Converts fraud detection results from different models into Excel format for analysis and comparison.

### Multi-Model Evaluation

- **run_all_model.py**: Runs evaluations for multiple models in parallel.
- **role_result.py**: Compares and summarizes the performance of different models in fraud detection tasks.

## Experiments and Evaluation

### Evaluation Metrics

- **Defense Success Rate (DSR)**: Measures the proportion of times the agent successfully resists fraudulent manipulation.
- **Precision@K (P@K)**: Assesses the ranking quality by counting how many factually correct options appear within the top-K positions.
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Rewards agents for placing authentic options closer to the top of the ranking.

### Experimental Setup

- **Multi-Agent Travel Planning Framework**: A framework simulating the entire travel planning process, including information retrieval, data extraction, tentative summary, and confirmation of plans.

## Getting Started

### Environment Configuration

1. Ensure Python and necessary libraries are installed.
2. Configure the `OPENAI_API_KEY` environment variable to access LLM services.
3. Adjust website lists and agent configurations in `trueinfo_gene.py` as needed.

### Running the Project

1. Clone the repository:
   ```
   git clone https://github.com/JhengXu/WandaPlan.git
   cd WandaPlan
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Collect real information:
   ```
   python trueinfo_gene.py
   ```

