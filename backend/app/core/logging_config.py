import logging
import sys
import json
import os
import logging.config

def setup_logging():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    possible_paths = [
        os.path.join(base_dir, "..", "config", "logging.json"),
        os.path.join(os.getcwd(), "config", "logging.json"),
        "config/logging.json",
        "../config/logging.json"
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
            
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        if "handlers" in config and "file" in config["handlers"]:
            project_root = os.path.dirname(base_dir)
            log_file = os.path.join(project_root, "metrics_history", "app.log")
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            config["handlers"]["file"]["filename"] = log_file

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config not found in {possible_paths}, using default.")

    logger = logging.getLogger(__name__)
    logger.info("Logging configured.")
