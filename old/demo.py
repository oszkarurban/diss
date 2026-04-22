import requests
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

messages = [{"role": "user", "content": "Think step by step. Explain speculative decoding."}]

input_ids = tok.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)

print("token ids:", input_ids)
print("num tokens:", len(input_ids))

data = {
    "input_ids": input_ids,  # pass token ids directly, not text
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": 200,
    },
}

response = requests.post("http://localhost:30000/generate", json=data)
result = response.json()

print(result["meta_info"])




#OPENAI

# url = f"http://localhost:{30000}/v1/chat/completions"

# data = {
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",#"Qwen/Qwen3-30B-A3B",#"meta-llama/Llama-3.1-8B-Instruct",#"Qwen/Qwen3-1.7B",
#     "messages": [{"role": "user", "content": "Think step by step. Explain speculative decoding."}],
# }

# response = requests.post(url, json=data)
# print(response.json())

# print(json.dumps(response.json()["meta_info"], indent=2))




##MANUAL
# url = "http://localhost:30000/generate"

# data = {
#     "text": "<｜User｜>Think step by step. Explain speculative decoding.<｜Assistant｜><think>\n", #dont add <｜begin▁of▁sentence｜>at the very beggining before User
#     "sampling_params": {
#         "temperature": 0,
#         "max_new_tokens": 208,
#     },
# }

# response = requests.post(url, json=data)
# result = response.json()

# # print(result)
# print(result["meta_info"])
# # print(json.dumps(result["meta_info"], indent=1))

