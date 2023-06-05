from itertools import cycle, product
import json
import random
import string
import pandas as pd

import requests

contexts = {
    # "dsail2": "ssh://henry@dsail2.cs.dartmouth.edu",
    # "default": "unix:///var/run/docker.sock",
    "mms-large-1": "unix:///var/run/docker.sock"
    # "mms-large-2": "ssh://henry@mms-large-2.cs.dartmouth.edu",
}

models = [
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

datasets = [
    "stereoset",
    "winobias",
    "crows_pairs"
]

training_types = [
    "classifieronly",
    "finetuned"
]

gpu_cards = [
    ("mms-large-1", 0),
    # ("mms-large-2", 0),
    # ("dsail2", 0),
    ("mms-large-1", 1),
    # ("mms-large-2", 1),
    # ("dsail2", 1),
    ("mms-large-1", 2),
    # ("mms-large-2", 2),
    # ("dsail2", 2),
    ("mms-large-1", 3),
    # ("mms-large-2", 3),
    # ("dsail2", 3),
    # ("mms-large-1", 4),
    # ("mms-large-2", 4),
    ("mms-large-1", 5),
    # ("mms-large-2", 5),
    ("mms-large-1", 6),
    # ("mms-large-2", 6),
    ("mms-large-1", 7),
    # ("mms-large-2", 7),
]


config = dict()
config["contexts"] = contexts
config["experiments"] = []

configs = product(models, datasets, training_types)

results = pd.read_csv("./results.csv")

def config_filter(config):
    model, dataset, training_type = config
    name = f"{model.replace('/', '-')}_{dataset}_{training_type}"
    acc = results.loc[results["name"] == name]["accuracy"].iloc[0]
    return not (acc >= 0.7)

good_configs = filter(config_filter, configs)

for idx, ((model, dataset, training_type), (context, card)) in enumerate(zip(good_configs, cycle(gpu_cards))):
    rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    config["experiments"].append({
      "name": f"{idx}_{model.replace('/', '-')}_{dataset}_{training_type}",
      "image": "ghcr.io/henryscheible/train:7de4ffb59211026fa52b1fe9fb051ed90ede8010",
      "context": context,
      "card": card,
      "buildargs": {
        "MODEL": model,
        "DATASET": dataset,
        "TRAIN_TYPE": training_type,
        "MODEL_TYPE": "generative",
        "LEARNING_RATE": 5e-4,
        "EPOCHS": 50,
        "SEED": 12345
      }
    })

with open("training_remnants.json", "w") as f:
    f.write(json.dumps(config))
