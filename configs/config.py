import yaml

def get_config(path: str):
    """
    Charge un fichier YAML de configuration et le retourne sous forme de dict Python.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
