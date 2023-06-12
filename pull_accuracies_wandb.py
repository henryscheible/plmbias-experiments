import pandas as pd
from tqdm import tqdm
import requests
import sys
import json
import wandb
from itertools import product
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

api = wandb.Api()

results = pd.DataFrame(columns=["name", "accuracy", "versions", "acc>0.7", "acc>0.75", "has_contribs", "contribs_artifact", "contribs_run_id", "has_encoder_ablation", "has_decoder_ablation", "encoder_ablation_artifact", "decoder_ablation_artifact", "has_encoder_ablation_vis", "has_decoder_ablation_vis", "encoder_ablation_vis_artifact", "decoder_ablation_vis_artifact"])

with open("specfile.json", "r") as specfile:
    raw_specs = json.loads(specfile.read())

specs = [f"{model.replace('/', '-')}_{dataset}_{train_type}" for model, train_type, dataset in product(raw_specs["models"], raw_specs["train_types"], raw_specs["datasets"])]

for spec in tqdm(specs):
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
    contrib_runs = list(filter(lambda run: "contribs" in run.name, best_model.used_by()))
    assert len(contrib_runs) <= 1
    if len(contrib_runs) > 0:
        contrib_run_id = contrib_runs[0].id
    else:
        contrib_run_id = None

    contrib_artifacts = [artifact for run in contrib_runs for artifact in list(filter(lambda artifact: artifact.type == "contribs", run.logged_artifacts()))]
    assert len(contrib_artifacts) <= 1
    has_contribs = False
    contribs_artifact = None
    if len(contrib_artifacts) > 0:
        has_contribs = True
        contribs_artifact = contrib_artifacts[0].name
        contribs_artifact_obj = contrib_artifacts[0]
    for version in model_versions:
        version.save()
    
    has_encoder_ablation = False
    has_decoder_ablation = False
    encoder_ablation_artifact = None
    decoder_ablation_artifact = None

    if has_contribs:
        ablation_candidates = [artifact for run in contribs_artifact_obj.used_by() for artifact in run.logged_artifacts()]
        encoder_ablation_candidates =  list(filter(lambda artifact: "latest" in artifact.aliases and "encoder" in artifact.name, ablation_candidates))
        decoder_ablation_candidates =  list(filter(lambda artifact: "latest" in artifact.aliases and "decoder" in artifact.name, ablation_candidates))
        assert len(encoder_ablation_candidates) <= 1
        assert len(decoder_ablation_candidates) <= 1
        if len(encoder_ablation_candidates) > 0:
            has_encoder_ablation = True
            encoder_ablation_artifact = encoder_ablation_candidates[0].name
            encoder_ablation_artifact_obj = encoder_ablation_candidates[0]
        if len(decoder_ablation_candidates) > 0:
            has_decoder_ablation = True
            decoder_ablation_artifact = decoder_ablation_candidates[0].name
            decoder_ablation_artifact_obj = decoder_ablation_candidates[0]

    has_encoder_ablation_vis = False
    encoder_ablation_vis_artifact = None

    has_decoder_ablation_vis = False
    decoder_ablation_vis_artifact = None

    if has_encoder_ablation:
        vis_candidates = [artifact for run in encoder_ablation_artifact_obj.used_by() for artifact in run.logged_artifacts()]
        vis_candidates =  list(filter(lambda artifact: "latest" in artifact.aliases, vis_candidates))
        if len(vis_candidates) > 0:
            has_encoder_ablation_vis = True
            encoder_ablation_vis_artifact = vis_candidates[0].name

    if has_decoder_ablation:
        vis_candidates = [artifact for run in decoder_ablation_artifact_obj.used_by() for artifact in run.logged_artifacts()]
        vis_candidates =  list(filter(lambda artifact: "latest" in artifact.aliases, vis_candidates))
        if len(vis_candidates) > 0:
            has_decoder_ablation_vis = True
            decoder_ablation_vis_artifact = vis_candidates[0].name

    
    results.loc[len(results)] = {
        "name": spec,
        "accuracy": max_acc,
        "acc>0.7": max_acc > 0.7,
        "acc>0.75": max_acc > 0.75,
        "versions": len(model_versions),
        "has_contribs": has_contribs,
        "contribs_artifact": contribs_artifact,
        "contribs_run_id": contrib_run_id,
        "has_encoder_ablation": has_encoder_ablation,
        "has_decoder_ablation": has_decoder_ablation,
        "encoder_ablation_artifact": encoder_ablation_artifact,
        "decoder_ablation_artifact": decoder_ablation_artifact,
        "has_encoder_ablation_vis": has_encoder_ablation_vis,
        "has_decoder_ablation_vis": has_decoder_ablation_vis,
        "encoder_ablation_vis_artifact": encoder_ablation_vis_artifact,
        "decoder_ablation_vis_artifact": decoder_ablation_vis_artifact
    }

results["versions"] = results["versions"].astype("Int64")
results["acc>0.7"] = results["acc>0.7"].astype("Int64").fillna(0)
results["acc>0.75"] = results["acc>0.75"].astype("Int64").fillna(0)

results.to_csv("results.csv")
print(results)
print(f"ACC > .7: {results['acc>0.7'].sum()}/{len(results)}")
print(f"ACC > .75: {results['acc>0.75'].sum()}/{len(results)}")
print(f"Probing Started: {results['contribs_run_id'].notnull().sum()}/{len(results)}")
print(f"Probing Finished: {results['has_contribs'].sum()}/{len(results)}")
print(f"Encoder Ablation Finished: {int(results['has_encoder_ablation'].sum())}/{len(results)}")
print(f"Decoder Ablation Finished: {int(results['has_decoder_ablation'].sum())}/{len(results)}")


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


