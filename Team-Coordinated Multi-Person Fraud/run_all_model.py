from multiprocessing import Process
from multi_flight import run_with_model  ##you can change “role_flight” to our else function

model_list = [
    "gpt-4o",
    "TA/mistralai/Mixtral-8x22B-Instruct-v0.1"
    "TA/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-3.5-turbo",
    "grok-3-beta",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest"
]

if __name__ == "__main__":
    processes = []
    for model in model_list:
        p = Process(target=run_with_model, args=(model,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
