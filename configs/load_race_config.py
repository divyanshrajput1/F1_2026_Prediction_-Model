import yaml

def load_race_config(path="configs/race_config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
