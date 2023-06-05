from itertools import cycle, product
import json
import random
import string

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
    # "t5-large",
    # "google/flan-t5-base",
    # "google/flan-t5-large"
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

for idx, ((model, dataset, training_type), (context, card)) in enumerate(zip(configs, cycle(gpu_cards))):
    rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    config["experiments"].append({
      "name": f"{idx}_{model.replace('/', '-')}_{dataset}_{training_type}",
      "image": "ghcr.io/henryscheible/train:82e6039e615f4c22986dc074e7d3970195e39553",
      "context": context,
      "card": card,
      "buildargs": {
        "MODEL": model,
        "DATASET": dataset,
        "TRAIN_TYPE": training_type,
        "MODEL_TYPE": "generative",
        "LEARNING_RATE": 0
      }
    })

with open("training_generative.json", "w") as f:
    f.write(json.dumps(config))
