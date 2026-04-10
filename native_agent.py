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

def search_wikipedia(query):
    """Search Wikipedia and return a summary."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return json.dumps({
                "title": data.get("title", ""),
                "summary": data.get("extract", "No summary found.")[:800]
            })
        return json.dumps({"error": f"Page not found for '{query}'. Try a different term."})
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculate_math(expression):
    """Safely evaluate a math expression."""
    try:
        allowed = set("0123456789+-*/.() eE")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": f"Invalid characters in: {expression}"})
        result = eval(expression)
        return json.dumps({"expression": expression, "result": round(result, 6)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_current_date():
    """Get the current date and time."""
    from datetime import datetime
    return json.dumps({"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression like '(5+3)*2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

NATIVE_FNS = {
    "search_wikipedia": search_wikipedia,
    "calculate": calculate_math,
    "get_current_date": get_current_date,
}

def run_native_agent(user_query, max_iterations=10, verbose=True):
    """
    Agent using native function calling.
    No regex — the API returns structured tool_calls.
    """
    messages = [
        {"role": "system", "content":
            "You are a helpful research assistant. Use tools to find information. "
            "Think step by step. After gathering enough info, give a complete answer."},
        {"role": "user", "content": user_query},
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f" USER: {user_query}")
        print(f"{'='*60}")

    input_token=0
    output_token=0

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1} ---")

        response = client.chat.completions.create(
            model=TOOL_MODEL, messages=messages,
            tools=NATIVE_TOOLS, temperature=0, max_tokens=800,
        )

        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        if finish == "stop" and msg.content:
            if verbose:
                print(f"\n{'='*60}")
                print(f"\n Total input tokens: {input_token}, Total output tokens: {output_token}, Total tokens: {input_token+output_token}")
                print(f"\n{'='*60}")  
                print(f" FINAL ANSWER:\n  {msg.content[:500]}")
            
            return {"answer": msg.content, "iterations": i + 1}

        # Tool calls
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                if verbose:
                    print(f" TOOL: {fn_name}({json.dumps(fn_args)[:80]})")

                if fn_name in NATIVE_FNS:
                    result = NATIVE_FNS[fn_name](**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                if verbose:
                    print(f" RESULT: {result[:150]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            if msg.content:
                messages.append({"role": "assistant", "content": msg.content})
            else:
                break

    print(f"\n{'='*60}")
    print(f"\n Total input tokens: {input_token}, Total output tokens: {output_token}, Total tokens: {input_token+output_token}")
    print(f"\n{'='*60}")   
    
    return {"answer": "Max iterations reached", "iterations": max_iterations}


result = run_native_agent(
    "What is mediasoup and who is the lead maintainer?"
)