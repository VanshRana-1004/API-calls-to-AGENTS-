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

def get_developer_info(name: str):
    """Get detailed developer profile including top projects and README summaries"""

    try:
        query = name.replace(" ", "+")
        search_url = f"https://api.github.com/search/users?q={query}"

        res = requests.get(search_url).json()
        if not res.get("items"):
            return json.dumps({"error": "No users found"})

        user = res["items"][0]
        username = user["login"]

        # Get user details
        user_data = requests.get(f"https://api.github.com/users/{username}").json()

        # Get repos
        repos = requests.get(user_data["repos_url"]).json()
        repos = sorted(repos, key=lambda x: x["stargazers_count"], reverse=True)

        top_repos = []

        for repo in repos[:2]:  # top 2 repos
            repo_name = repo["name"]

            # Fetch README
            readme_text = ""
            try:
                readme_res = requests.get(
                    f"https://api.github.com/repos/{username}/{repo_name}/readme"
                ).json()

                if "content" in readme_res:
                    readme_text = base64.b64decode(readme_res["content"]).decode("utf-8")[:1000]
            except:
                readme_text = "README not available"

            top_repos.append({
                "name": repo_name,
                "stars": repo["stargazers_count"],
                "url": repo["html_url"],
                "description": repo["description"],
                "readme_snippet": readme_text
            })

        return json.dumps({
            "username": username,
            "bio": user_data.get("bio"),
            "followers": user_data.get("followers"),
            "top_projects": top_repos
        })

    except Exception as e:
        return json.dumps({"error": str(e)})

def get_current_date():
    """Get the current date and time."""
    from datetime import datetime
    return json.dumps({"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

TOOLS = [
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
        "type":"function",
        "function":{
            "name" : "get_developer_info",
            "description":"Get Github profile info for a developer. Example input: {'name': 'VanshRana-1004'}",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Developer's name"}
                },
                "required": ["name"]
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

FNS = {
    "search_wikipedia": search_wikipedia,
    "calculate": calculate_math,
    "get_developer_info": get_developer_info,
    "get_current_date": get_current_date,
}

def run_agent(user_query, max_iterations=15, verbose=True):

    messages=[
        {"role":"system", "content": "you are a helpful research assistant. Use tools to find information. Think step by step. After gathering enough info, give a complete answer. "},
        {"role":"user", "content": user_query}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"User : {user_query}")
        print(f"\n{'='*60}")

    input_token=0
    output_token=0
    
    for i in range(max_iterations):

        if verbose:
            print(f"\n --- Iteration {i+1}/{max_iterations} ---")

        if((i+1)%2==0):
            # summarize 
            summarize=client.chat.completions.create(
                model=FREE_MODEL,
                messages=messages + [{"role":"user", "content": "Summarize the conversation so far in 2-3 sentences. Focus on key info and tools used. This will help you keep track of the context."}],
                temperature=0,
                max_tokens=150
            )

            input_token+=summarize.usage.prompt_tokens
            output_token+=summarize.usage.completion_tokens

            summary_text=summarize.choices[0].message.content or ""
            new_messages=[]
            
            if messages[0]["role"]=="system":
                new_messages.append(messages[0])

            new_messages.append({"role":"system", "content": summary_text})

            print(f"\nSummary after {i+1} iterations:\n{summary_text}\n")

            messages=new_messages
        
        res=client.chat.completions.create(
            model=TOOL_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0,
            max_tokens=800
        )

        input_token+=res.usage.prompt_tokens
        output_token+=res.usage.completion_tokens

        msg=res.choices[0].message
        finish=res.choices[0].finish_reason

        if finish=="stop" and msg.content:
            if verbose:
                print(f"\n{'='*60}")
                print(f"\n Total input tokens: {input_token}, Total output tokens: {output_token}, Total tokens: {input_token+output_token}")
                print(f"\n{'='*60}")   

                print(f"FINAL ANSWER: {msg.content}")

            return {"answer": msg.content, "iterations": i+1}
        
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                if verbose:
                    print(f" TOOL: {fn_name}({json.dumps(fn_args)[:80]})")

                if fn_name in FNS:
                    result = FNS[fn_name](**fn_args)
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

result=run_agent(
    "What is mediasoup and who is the lead maintainer?"    
)