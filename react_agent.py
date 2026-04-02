import os
import json
import re
import time
import requests
import base64
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


# 1) writing system prompt
REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step using tools.

You have access to these tools:
{tool_descriptions}

## How to respond

When you need to use a tool, respond in EXACTLY this format:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <arguments as valid JSON>

When you have enough information for the final answer:

THOUGHT: <your final reasoning>
FINAL_ANSWER: <your complete answer to the user>

## Rules
- Always start with THOUGHT
- Use only ONE action per turn
- Wait for the OBSERVATION before continuing
- If a tool returns an error, reason about it and try a different approach
- Be concise in your thoughts
"""


# 2) Writing tool functions 
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

# Tool registry
TOOLS = {
    "search_wikipedia": {
        "fn": search_wikipedia,
        "desc": "search_wikipedia(query: str) — Search Wikipedia. Use simple topic names like 'France' or 'Albert Einstein'."
    },
    "dev_info":{
        "fn": get_developer_info,
        "desc": "dev_info(name: str) — Get  GitHub search URLs for a person. Example: 'Vansh Rana'"
    },
    "calculate": {
        "fn": calculate_math,
        "desc": "calculate(expression: str) — Evaluate a math expression. Example: '(5 + 3) * 2.5'"
    },
    "get_current_date": {
        "fn": get_current_date,
        "desc": "get_current_date() — Get today's date and time. Pass empty JSON: {}"
    },
}

# print(f"Tools registered: {list(TOOLS.keys())}")


# 3) Thought -> Action -> (Observe -> Thought Loop) / Final Answer

def run_agent(user_query, max_iterations=10, verbose=True):

    tool_desc = "\n".join(f"-{t['desc']}" for t in TOOLS.values())
    system = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    messages = [
        {"role" : "system", "content" : system},
        {"role" : "user", "content" : user_query}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"User : {user_query}")
        print(f"\n{'='*60}")

    for i in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")
        
        # Step 1 : ask the LLM what to do
        res=client.chat.completions.create(
            model=FREE_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=800
        )

        text = res.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        # Step 2 : parse the response Check for final answer
        thought_match = re.search(r"THOUGHT:\s(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        if thought : 
            print(f"Thought: {thought}")

        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER : ")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"AGENT FINISHED in {i+1} iteration(s)")
                print(f"{'='*60}")
                print(f"ANSWER: {answer}")
            return {"answer": answer, "iterations": i + 1}

        action_match=re.search(r"ACTION:\s*(\w+)",text)
        input_match=re.search(r"ACTION_INPUT:\s*(.+?)(?:\n|$)",text,re.DOTALL)

        if not action_match:
            messages.append({
                "role": "user",
                "content": "Please respond with either ACTION + ACTION_INPUT or FINAL_ANSWER."
            })
            if verbose:
                print(" Format issue — nudging...")
            continue

        tool_name = action_match.group(1).strip()
        raw_input = input_match.group(1).strip() if input_match else "{}"

        # Step 3 : Execute the tool action
        if tool_name not in TOOLS:
            observation = json.dumps({
                "error": f"Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
            })
        else:
            try:
                if raw_input.startswith("{"):
                    args = json.loads(raw_input)
                else:
                    args = {"query": raw_input.strip("\"'")}
                observation = TOOLS[tool_name]["fn"](**args)
            except Exception as e:
                observation = json.dumps({"error": f"Failed to call {tool_name}: {e}"})

        if verbose:
            print(f"ACTION: {tool_name}({raw_input})")
            obs_preview = observation + "..." if len(observation) > 200 else observation
            print(f"OBSERVATION: {obs_preview}")

        # STEP 4: Feed observation back
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    if verbose:
        print(f"\n Max iterations ({max_iterations}) reached.")
    return {"answer": "Max iterations reached.", "iterations": max_iterations}

# funtion call
# result = run_agent("What is 123*123?")

result = run_agent(
    "Who is Devansh Sabharwal? (developer)"
)