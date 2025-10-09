import os
from pathlib import Path
import logging 

logging.basicConfig(level = logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "stored_index.py",
    "static/.gitkeep",
    "templates/chat.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"Creating directory; {filedir} for the file {filename} ")

    if not os.path.exists(filepath):
        # File doesn't exist, create it
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    elif os.path.getsize(filepath) == 0:
        # File exists but is empty
        logging.info(f"File {filepath} exists but is empty")
    else:
        # File exists and has content
        logging.info(f"{filename} already exists")
    

