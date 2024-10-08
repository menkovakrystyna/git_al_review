import os
import subprocess
from openai import OpenAI

# Initialize OpenAI client with the API key from the environment
with open('api_key.txt') as file:
    val = file.readline().strip()

# Initialize OpenAI client
client = OpenAI(api_key=val)

# Function to get the modified/added files from the latest commit
def get_modified_files():
    # Run git command to list modified and added files in the last commit
    result = subprocess.run(['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'], capture_output=True, text=True)
    files = result.stdout.splitlines()
    return [file for file in files if file.endswith(('.py', '.js', '.java', '.html', '.css'))]

# Function to read code files
def read_code_files(files):
    code_content = ""
    for file_path in files:
        with open(file_path, 'r') as f:
            code_content += f"\n\n{file_path}:\n"
            code_content += f.read()
    return code_content

# Get the list of modified/added files
modified_files = get_modified_files()

# Read the content of the modified files
code_content = read_code_files(modified_files)

# Send the code to GPT for review if there are modified files
if code_content:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a code reviewer."},
            {"role": "user", "content": f"Please review the following code:\n{code_content}"}
        ]
    )

    # Output GPT's response
    print(completion.choices[0].message)
else:
    print("No code changes detected.")
