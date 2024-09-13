import os
import subprocess
from openai import OpenAI

# Access the environment variable
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("No API key found. Set the OPENAI_API_KEY environment variable.")

# Change working directory to the root of the repository
repo_root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
os.chdir(repo_root)

# Function to get the modified/added files by comparing the new branch to the main branch
import subprocess

import subprocess

import subprocess


import subprocess

def get_modified_files():
    # Using `git show` to get the files changed in the latest commit
    result = subprocess.run(['git', 'show', '--name-only', 'HEAD'], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running git show")
        return []

    print(f"git show output: {result.stdout}")  # Debug line to show file changes
    files = result.stdout.splitlines()

    # Filter the files based on your desired extensions
    return [file for file in files if file.endswith(('.py', '.js', '.java', '.html', '.css'))]

# Example usage
modified_files = get_modified_files()
print("Modified files:", modified_files)


# Function to read the content of modified code files
def read_code_files(files):
    code_content = ""
    for file_path in files:
        with open(file_path, 'r') as f:
            code_content += f"\n\n{file_path}:\n"
            code_content += f.read()
    return code_content

# Get the list of modified/added files by comparing the current branch with main
modified_files = get_modified_files()

# Read the content of the modified files
code_content = read_code_files(modified_files)

# Send the code to GPT for review if there are modified files
if code_content:
    client = OpenAI(api_key=api_key)
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
