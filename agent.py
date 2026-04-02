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


def read_file(path):
    try:
        with open(path,"r") as f:
            content=f.read()
            return json.dumps({"path":path, "content": content, "truncated":len(content)>3000})
    except Exception as e:
        return json.dumps({"error": str(e)})

def write_file(path, content):
    try:
        d=os.path.dirname(path)
        if d:
            os.makedirs(d,exist_ok=True)
        with open(path,"w") as f:
            f.writ(content)
        return json.dumps({"status":"success","path":path,"bytes":len(content)})
    except Exception as e:
        return json.dumps({"error": str(e)})

def run_python(code):
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        return json.dumps({
            "stdout": result.stdout[:2000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Timed out (15s limit)"})
    except Exception as e:
        return json.dumps({"error": str(e)})

def list_files(path="."):
    try:
        items = sorted(os.listdir(path))
        return json.dumps({"path": path, "files": items[:50]})
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

CODE_TOOLS = {
    "read_file": {"fn": read_file, "desc": "read_file(path: str) — Read a file and return its contents."},
    "write_file": {"fn": write_file, "desc": 'write_file(path: str, content: str) — Write content to a file.'},
    "run_python": {"fn": run_python, "desc": "run_python(code: str) — Execute Python code. Returns stdout/stderr."},
    "list_files": {"fn": list_files, "desc": "list_files(path: str) — List files in a directory."},
    "calculate": {"fn": calculate_math, "desc": "calculate(expression: str) — Evaluate a math expression."},
}

print(f"Code tools: {list(CODE_TOOLS.keys())}")

CODE_SYSTEM_PROMPT = """You are a coding agent. You write, run, and debug Python code to solve tasks.

Available tools:
{tool_descriptions}

Format — to use a tool:

THOUGHT: <your reasoning>
ACTION: <tool_name>
ACTION_INPUT: {{"arg1": "value1", "arg2": "value2"}}

Format — when done:

THOUGHT: <final reasoning>
FINAL_ANSWER: <your answer>

## Rules
- ONE action per turn. Wait for OBSERVATION.
- After writing code to a file, always run_python to TEST it.
- If a test fails, read the error, fix the code, try again.
- Verify your work before giving FINAL_ANSWER.
- Include print() statements so you can see output.
"""

def run_code_agent(user_query, max_iterations=15, verbose=True):
    
    tool_desc="\n".join(f"- {t['desc']}" for t in CODE_TOOLS.values())
    system=CODE_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user_query}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK : {user_query}")
        print(f"{'='*60}")

    for i in range(max_iterations):

        if verbose:
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")

        res=client.chat.completions.create(
            model=FREE_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1500
        )

        text=res.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|FINAL_ANSWER:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        if thought:
            print(f"Thought: {thought}")
        
        if "FINAL_ANSWER:" in text:
            answer = text.split("FINAL_ANSWER:")[-1].strip()
            if verbose:
                print(f"\n{'='*60}")
                print(f"DONE in {i+1} iteration(s)")
                print(f"{'='*60}")
                print(f"FINAL ANSWER: {answer}")
            return answer
        
        action_match = re.search(r"ACTION:\s*(\w+)", text)  
        input_match = re.search(r"ACTION_INPUT:\s*(.+?)(?:\nTHOUGHT|\nACTION|\nFINAL)", text, re.DOTALL)

        if not action_match:
            messages.append({"role": "user", "content": "Use ACTION + ACTION_INPUT or FINAL_ANSWER."})
            if verbose:
                print(" Format issue — nudging...")
            continue

        tool_name=action_match.group(1).strip()
        raw_input=input_match.group(1).strip() if input_match else "{}"

        if verbose:
            print(f" ACTION : {tool_name}")

        if tool_name not in CODE_TOOLS:
            observation = json.dumps({"error" : f"Unknown tool '{tool_name}'. Available: {list(CODE_TOOLS.keys())}"}) 
        else:
            try:
                args=json.loads(raw_input)
                observation = CODE_TOOLS[tool_name]["fn"](**args)
            except json.JSONDecodeError:
                try:
                    observation = CODE_TOOLS[tool_name]["fn"](raw_input.strip("\"'"))
                except Exception as e:
                    observation = json.dumps({"error": f"Parse + fallback failed: {e}"})
            except Exception as e:
                observation = json.dumps({"error": str(e)})

        if verbose:
            obs_short = observation[:250] + "..." if len(observation)>250 else observation
            print(f" OBSERVATION : {obs_short}")
        
        messages.append({"role":"user", "content": f"OBSERVATION: {observation}"})

    return "\n [MAX ITERATIONS REACHED] \n"


result = run_code_agent(
    "Write a Python function called 'is_palindrome(n)' that checks whether the given string is palindrome or not. A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward (ignoring spaces, punctuation, and capitalization). After writing the function, test it with the string"
    "Save it to 'palidrome.py'. Then write a test script that imports it and tests with: "
    " [\"abcba\"], [\"palindrome\"], [\"HOLLALLOH\"], [\"VANSH\"], [\"HAHAHAHAHAHAHAHAHAHAH\"]"
    "Whenever taking any action, its your responsibility to provide respective path and content to the called function."
)

# result = run_code_agent(
#     "Write a Python function called 'is_prime(n)' that checks if a number is prime. "
#     "Save it to 'prime.py'. Then write a test script that imports it and tests with: "
#     "1, 2, 13, 15, 97, 100. Print the results."
# )

