from itertools import cycle, product
import json
import random
import string
import pandas as pd
import wandb
api = wandb.Api()

import requests

contexts = {
    "dsail2": "ssh://henry@dsail2.cs.dartmouth.edu",
    # "default": "unix:///var/run/docker.sock",
    # "mms-large-1": "unix:///var/run/docker.sock"
    # "mms-large-2": "ssh://henry@mms-large-2.cs.dartmouth.edu",
}

gpu_cards = [
    # ("mms-large-1", 0),
    # ("mms-large-2", 0),
    ("dsail2", 0),
    # ("mms-large-1", 1),
    # ("mms-large-2", 1),
    ("dsail2", 1),
    # ("mms-large-1", 2),
    # ("mms-large-2", 2),
    ("dsail2", 2),
    # ("mms-large-1", 3),
    # ("mms-large-2", 3),
    ("dsail2", 3),
    # ("mms-large-1", 4),
    # ("mms-large-2", 4),
    # ("mms-large-1", 5),
    # ("mms-large-2", 5),
    # ("mms-large-1", 6),
    # ("mms-large-2", 6),
    # ("mms-large-1", 7),
    # ("mms-large-2", 7),
]


config = dict()
config["contexts"] = contexts
config["experiments"] = []

results = pd.read_csv("./results.csv")

contribs_artifacts = results["contribs_artifact"].dropna().unique().tolist()
datasets = ["_".join(artifact.split("_")[1:-2]) for artifact in contribs_artifacts]
specs = zip(contribs_artifacts, datasets)
portioned_specs = product(specs, ["encoder", "decoder"])
configs = [(contribs, dataset, portion) for (contribs, dataset), portion in portioned_specs]

print(datasets)

for idx, ((contribs, dataset, portion), (context, card)) in enumerate(zip(configs, cycle(gpu_cards))):
    rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    config["experiments"].append({
      "name": f"{idx}_{contribs.replace('/', '-').replace(':', '-')}_{dataset}_{portion}_ablation",
      "image": "ghcr.io/henryscheible/ablation:980be76b9ed83ee78621c01ee505c6b65080d89e",
      "context": context,
      "card": card,
      "buildargs": {
        "CONTRIBS": contribs,
        "DATASET": dataset,
        "PORTION": portion
      }
    })

with open("ablation.json", "w") as f:
    f.write(json.dumps(config))
