import os
import hashlib
import subprocess
from openai import OpenAI
from typing import Optional
from constants import ACCESS
import regex as re

client = OpenAI(api_key=ACCESS )  


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
                    
                    "if user query is related to car information or country information, use the existing data"
                    "Use the raw CSV located at ../data/raw/cars_dataset.csv for structured car data."
                    "For unstructured country data, refer to data/raw/country_data.md."
                    "Do not use the processed/ folder for parsing — it's derived."
                    "The dataset is a CSV file with the following columns:"

                    "Car Name: The model name of the car. (e.g., 'OffDecision12')"

                    "Manufacturer: The brand or company that manufactures the car. (e.g., 'Ross PLC')"

                    "Launch Year: The year the car was released. (e.g., 2017)"

                    "Description: A textual summary or marketing description of the car."

                    "Engine Specifications: A string describing the engine type, horsepower, and engine capacity. Example format: V6, 422 HP, 1762cc"

                    "Other Specifications: A string describing the body type, mileage, and top speed. Example format: SUV, 10 km/l, 203 km/h top speed"

                    "User Ratings: A float value representing the average user rating out of 5. (e.g., 2.5)"
                    "NCAP Global Rating: An integer from 1 to 5 representing the safety rating from NCAP."
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
        [r"C:\Users\infinix\OneDrive\Desktop\Amit_Pandey\venv\ragbot312\Scripts\python.exe", file_path],
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