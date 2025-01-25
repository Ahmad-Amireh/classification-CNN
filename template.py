import os 
from pathlib import Path 
import logging 


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:|%(message)s:')
project_name = "mlproject"

list_of_files = [
    ".github/workflows/.gitkeep",  # CI/CD
    f"src/{project_name}/__init__.py", #package
    f"src/{project_name}/components/__init__.py", # 
    f"src/{project_name}/utils/__init__.py", 
    f"src/{project_name}/utils/common.py", # for all functions that used more thatn once across all the project 
    f"src/{project_name}/config/__init__.py", 
    f"src/{project_name}/config/configuration.py", # for all configuration 
    f"src/{project_name}/pipeline/__init__.py", # for pipeline 
    f"src/{project_name}/entity/__init__.py", 
    f"src/{project_name}/entity/config_entity.py", # for each once (seperate function)
    f"src/{project_name}/constants/__init__.py", # for all Constants 
    "config/config.yaml", # config 
    "dvc.yaml",
    "params.yaml", # parameter 
    "schema.yaml", # data schema 
    "main.py", # main 
    "app.py", # app 
    "Dockerfile", 
    "requirments.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"]

for filepath in list_of_files: 
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file name: {filename}")
    
    if ((not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0)): #not exist or empty 
        with open (filepath, "w") as f :
            pass
            logging.info(f"Creating empty file: {filepath}")
        
    else: 
        logging.info(f"{filename} is already exists")


