import os 
from box.exceptions import BoxValueError
import yaml
from CNNproject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox :
    """ Read Yaml file and returns
    
        Args:
            path_to_yaml(str) : path like Input
            
        Raises: 
            ValueError: if yaml file is empty 
            e: empty file 
        
        Returns: 
            ConfigBox: ConfigBox type
    
    """

    try: 
        with open (path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loadded successfully")
            return ConfigBox (content)
    
    except BoxValueError :
        raise ValueError ("yaml file is empty")
    except Exception as e :
        raise e     
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories 

    Args:
        path_to_directories(list): list of path of directories
        ignore_log (bool, optional) : ignore if multiple dirs is to be created. Defaults to False. 
    """

    for path in path_to_directories : 
        os.makedirs(path, exist_ok=True)
        if verbose :
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json (path:Path, data:dict):
    """Save Json data
    
    Args:
        path (str): path to json file 
        data (dict): data to be saved in Json file
        """
    with open (path,"w") as f: 
        json.dump(data, f, indent=4)

    logger.info(f"Json file saved at: {[path]}")



@ensure_annotations
def losad_json (path:Path):
    """Save Json data
    
    Args:
        path (str): path to json file 
        """
    with open (path,"w") as f: 
        content = json.load(f)

    logger.info(f"Json file loadded successfully from : {[path]}")
    return ConfigBox(content)


@ensure_annotations
def save_bin (path:Path, data:Any):
    """Save binary file 
    
    Args:
        path (str): path to binary file 
        data (Any): data to be saved into binary
        """
    joblib.dump(value=data, filenaem=path)

    logger.info(f"binary file saved at : {[path]}")



@ensure_annotations
def losad_json (path:Path):
    """Save binary data
    
    Args:
        path (str): path to binary file 
        """
    with open (path,"w") as f: 
        data = joblib.load(f)

    logger.info(f"Binary file loadded successfully from : {[path]}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in Kb 
    
    Args:
        path(str) : path of the file
        
    Return:
        str: size in Kb"""
    size_in_Kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_Kb} KB"

@ensure_annotations
def decodeImage(imgString, filename):
    img_data = base64.b64decode(imgString)
    with open(filename, "wb") as f:
        f.write(img_data)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f :
        return base64.b64encode(f.read())