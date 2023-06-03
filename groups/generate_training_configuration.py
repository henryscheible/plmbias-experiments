from itertools import cycle, product
import json
import random
import string

import requests

contexts = {
    # "dsail2": "ssh://henry@dsail2.cs.dartmouth.edu",
    # "default": "unix:///var/run/docker.sock",
    "mms-large-1": "ssh://henry@mms-large-1.cs.dartmouth.edu"
    # "mms-large-2": "ssh://henry@mms-large-2.cs.dartmouth.edu",
}

models = [
    "t5-small",
    # "t5-base",
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
      "image": "ghcr.io/henryscheible/train:e7dc41f3c035a9669eae132a91f33708414f6f61",
      "context": context,
      "card": card,
      "buildargs": {
        "MODEL": model,
        "DATASET": dataset,
        "TRAIN_TYPE": training_type,
        "MODEL_TYPE": "generative",
        "LEARNING_RATE": 5e-5
      }
    })

with open("training_generative.json", "w") as f:
    f.write(json.dumps(config))
