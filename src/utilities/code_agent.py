import os
import hashlib
import subprocess
from openai import OpenAI
from typing import Optional
from constants import ACCESS
import regex as re

# OpenAI client setup
client = OpenAI(api_key=ACCESS )  


# Directory to store cached math scripts
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "math_scripts")
os.makedirs(SCRIPT_DIR, exist_ok=True)


def generate_python_code(prompt: str) -> str:
    """Generates clean Python code using OpenAI for a given prompt."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python programmer. Return only valid, executable Python code. "
                    "Do not use ```python or any Markdown formatting. No explanations or comments — only the code body."
                )
            },
            {
                "role": "user",
                "content": f"{prompt}"
            }
        ],
        temperature=0
    )

    raw_code = response.choices[0].message.content.strip()

    # Remove Markdown formatting 
    code = re.sub(r"^```(?:python)?\s*", "", raw_code)
    code = re.sub(r"\s*```$", "", code)

    return code


def get_hashed_filename(prompt: str) -> str:
    """Returns a hashed filename based on the prompt."""
    hashed = hashlib.sha256(prompt.encode()).hexdigest()[:10]
    return f"script_{hashed}.py"


def save_python_script(code: str, filename: str) -> str:
    """Saves Python code to a file and returns the path."""
    script_path = os.path.join(SCRIPT_DIR, filename)
    with open(script_path, "w") as f:
        f.write(code)
    return script_path


def execute_generated_code(file_path: str) -> str:
    """Executes the generated script and returns output and errors."""
    result = subprocess.run(
        ["python", file_path],
        capture_output=True,
        text=True
    )
    output = result.stdout.strip()
    errors = result.stderr.strip()
    print (f"Output:\n{output}\n\nErrors:\n{errors}" if errors else f"Output:\n{output}")


def run_code_agent(prompt: str) -> str:
    """Main entry for the math agent — handles caching, generation, execution."""
    filename = get_hashed_filename(prompt)
    script_path = os.path.join(SCRIPT_DIR, filename)

    if os.path.exists(script_path):
        print(f"Using cached script: {filename}")
    else:
        print(f"Generating new script for prompt: {prompt}")
        code = generate_python_code(prompt)
        save_python_script(code, filename)

    return execute_generated_code(script_path)
