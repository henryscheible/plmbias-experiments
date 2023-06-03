import pandas as pd
from tqdm import tqdm
import requests
import sys
import json
import wandb
from itertools import product
import numpy as np

api = wandb.Api()

results = pd.DataFrame(columns=["name", "accuracy", "versions", "has_contribs"])

with open("specfile.json", "r") as specfile:
    raw_specs = json.loads(specfile.read())

specs = [f"{model.replace('/', '-')}_{dataset}_{train_type}" for model, train_type, dataset in product(raw_specs["models"], raw_specs["train_types"], raw_specs["datasets"])]

for spec in specs:
    try:
        model_versions = list(api.artifact_versions("model", f"plmbias/model-{spec}"))
    except wandb.errors.CommError:
        results.loc[len(results)] = {"name": spec}
        continue
    accuracies = []
    for version in model_versions:
        acc = version.logged_by().summary["eval/accuracy"]["max"]
        version.metadata["eval/accuracy"] = acc
        accuracies += [acc]
    accuracies = np.array(accuracies)
    max_acc = np.max(accuracies)
    best_model_idx = np.argmax(accuracies)
    best_model = model_versions[best_model_idx]
    best_model.aliases.append('best')
    for version in model_versions:
        version.save()
    results.loc[len(results)] = {
        "name": spec,
        "accuracy": max_acc,
        "versions": len(model_versions)
    }
results["versions"] = results["versions"].astype("Int64")
print(results)

# for launch_file in sys.argv[1:]:
#     with open(launch_file, "r") as file:
#         data = json.loads("".join(file.readlines()))
#     results["name"] = [experiment["name"] for experiment in data["experiments"]]
#     results = results.set_index("name")
#     for experiment in tqdm(data["experiments"]):
#         try:
#             validation = requests.get(f"https://huggingface.co/henryscheible/{experiment['name']}/raw/main/README.md").text
#             val_lines = validation.split("\n")
#             acc_line = list(filter(lambda l: "Accuracy:" in l, val_lines))[0]
#             acc = acc_line[12:]
#             results.loc[experiment["name"]]["accuracy"] = acc
#         except:
#             pass
#         try:
#             contribs = requests.get(
#                 f"https://huggingface.co/henryscheible/{experiment['name']}/raw/main/contribs.txt").text
#             if contribs[0] == "[":
#                 results.loc[experiment["name"]]["has_contribs"] = True
#         except:
#             pass
#     print(results)


