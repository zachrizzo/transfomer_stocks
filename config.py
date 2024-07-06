# config.py
import json
from datetime import datetime

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'hidden_size': 512,
            'num_layers': 3,
            'num_heads': 0,
            'dropout': 0.1,
            'num_epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.001,
            'start_date': datetime(2018, 1, 1).date(),
            'end_date': datetime(2024, 1, 1).date()
        }

def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)
