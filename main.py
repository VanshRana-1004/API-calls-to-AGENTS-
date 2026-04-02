import os
import json
import re
import time
import requests
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

FREE_MODEL="openrouter/free"
TOOL_MODEL="openrouter/free"

# if client is None:
#     print("Error: OpenAI client not initialized. Please check your API key.")
#     exit(1)
# else:
#     print("OpenAI client initialized successfully.")

# response = client.chat.completions.create(
#     model=FREE_MODEL,
#     messages=[{"role":"user", "content":"What is mediasoup SFU?"}],
#     temperature=0.0,
#     max_tokens=100 
# )

# print("Response : ", response.choices[0].message.content)
# print(f"Tokens - in : {response.usage.prompt_tokens}, out: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
# print(f"Model Used : {response.model}")

# prompt = "give me a quote regarding life"

# for temp in [0.0, 1.0, 2.0]:
#     print(f"\nTemperature: {temp}")
#     for _ in range(3):
#         r=client.chat.completions.create(
#             model=FREE_MODEL,
#             messages=[{"role" : "user", "content": prompt}],
#             temperature=temp,
#             max_tokens=60
#         )
#         res=r.choices[0].message.content
#         if res:
#             print(f"\nQuote: {res.strip()}")
#         else: 
#             print(f"\nNo quote generated.")
        
#     print(f"-" * 50)

# conversation = [
#     {"role" : "system", "content" : "You are a math tutor. Be concise"},
#     {"role" : "user", "content" : "What is differentiation <3"}
# ]

# r1=client.chat.completions.create(
#     model=FREE_MODEL,
#     messages=conversation,
#     max_tokens=150
# )

# res1=r1.choices[0].message.content or "[No content returned]"
# print(f"Response 1: {res1.strip()}")
# print(f"Tokens - in : {r1.usage.prompt_tokens}, out: {r1.usage.completion_tokens}, total: {r1.usage.total_tokens}")

# conversation.append({"role":"assistant", "content":res1})
# conversation.append({"role":"user", "content":"Can you explain me how it is different from integration?"})

# r2=client.chat.completions.create(
#     model=FREE_MODEL,
#     messages=conversation,
#     max_tokens=150 
# )

# res2=r2.choices[0].message.content or "[No content returned]"
# print(f"Response 2: {res2.strip()}")
# print(f"Tokens - in : {r2.usage.prompt_tokens}, out: {r2.usage.completion_tokens}, total: {r2.usage.total_tokens}")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(5 + 3) * 2'"}
                },
                "required": ["expression"]
            }
        }
    }
]

r=client.chat.completions.create(
    model=TOOL_MODEL,
    messages=[{"role":"user", "content":"What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

res=r.choices[0].message
print(f"Reason : {r.choices[0].finish_reason}")
print(f"Content : {res.content}")
print(f"Tool calls : {res.tool_calls}")

if res.tool_calls:
    tc=res.tool_calls[0]
    print(f"\n Model wants to call : {tc.function.name}({tc.function.arguments})")


def get_weather(city):
    fake={
        "Tokyo" : {"temp" : "22°C", "conditions" : "partly cloudy"},
        "London": {"temp": "14°C", "condition": "rainy"},
        "Delhi": {"temp": "38°C", "condition": "sunny"},
    }
    return json.dumps(fake.get(city, {"temp": "N/A", "condition": "N/A"}))

def calculate(expression):
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Invalid characters"})
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})

TOOL_FNS={"get_weather": get_weather, "calculate": calculate}

if res.tool_calls:
    tc=res.tool_calls[0]
    fn = TOOL_FNS[tc.function.name]
    result = fn(**json.loads(tc.function.arguments))
    print(f"\nTool response : {result}")

    messages=[
        {"role":"user", "content":"What's the weather in Tokyo?"},
        res,
        {"role":"tool", "tool_call_id":tc.id, "content":result}
    ]

    follow_up=client.chat.completions.create(
        model=TOOL_MODEL,
        messages=messages,
        tools=tools,
    )

    print(f"\n Final answer : {follow_up.choices[0].message.content}")
