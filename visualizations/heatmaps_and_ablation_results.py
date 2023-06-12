import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb 
import os

os.environ["WANDB_SILENT"] = "true"

def generate_heatmap(contribs, mask_shape):
    sns.set(font_scale=1.2)
    matrix = np.flip(np.array(contribs).reshape(mask_shape))
    xticklabels = list(np.arange(matrix.shape[1]) + 1)
    yticklabels = list(matrix.shape[0] - np.arange(matrix.shape[0]))
    ax = plt.gca()
    sns.heatmap(matrix, ax=ax, cmap="PiYG", center=0, xticklabels=xticklabels, yticklabels=yticklabels, square=True, cbar=False, vmin=-1, vmax=1)
    ax.set(
        xlabel="Head",
        ylabel="Layer"
    )
    plt.savefig("heatmap.pdf", format="pdf")
    plt.close()

def generate_figure(results):
    sns.set(font_scale=1.2)
    df = pd.DataFrame({
        'heads': np.arange(len(results["contribs"]) + 1),
        'bottom-up': results["bottom_up_results"],
        'top-down': results["bottom_up_rev_results"],
    })
    melted_df = pd.melt(df, ['heads']).rename(columns={"variable": "Settings"})
    ax = sns.lineplot(x='heads', y='value', hue='Settings',
                data=melted_df)
    ax.set(xlabel="Number of Active Attention Heads", ylabel="Stereotype Detection Performance (Accuracy)")
    plt.legend(loc="upper left", title="Settings")
    plt.savefig(f"add_ablation.pdf", format="pdf")
    plt.close()

    df = pd.DataFrame({
        'heads': np.arange(len(results["contribs"])+1),
        'bottom-up': results["top_down_results"],
        'top-down': results["top_down_rev_results"],
    })
    melted_df = pd.melt(df, ['heads']).rename(columns={"variable": "Settings"})
    ax = sns.lineplot(x='heads', y='value', hue='Settings',
                data=melted_df)
    ax.set(xlabel="Number of Pruned Attention Heads", ylabel="Stereotype Detection Performance (Accuracy)")
    plt.legend(loc="upper right", title="Settings")
    plt.savefig(f"remove_ablation.pdf", format="pdf")
    plt.close()

api = wandb.Api()

results = pd.read_csv("../results.csv")

encoder_ablation_artifacts = results["encoder_ablation_artifact"].dropna().tolist()
decoder_ablation_artifacts = results["decoder_ablation_artifact"].dropna().tolist()

for artifact_name in tqdm(encoder_ablation_artifacts):
    run = wandb.init(project="plmbias", name=f"{artifact_name.split(':')[0]}_visualizations")
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    with open(os.path.join(artifact_dir, "results.json"), "r") as file:
        results = json.loads(file.read())
    with open(os.path.join(artifact_dir, "shapes.json"), "r") as file:
        shapes = json.loads(file.read())

    contribs = results["contribs"]
    generate_heatmap(contribs, shapes["encoder"])

    generate_figure(results)

    artifact = wandb.Artifact(
        name=f"{artifact.name.split(':')[0]}_visualization",
        type="ablation_visualization"
    )
    artifact.add_file("heatmap.pdf")
    artifact.add_file("add_ablation.pdf")
    artifact.add_file("remove_ablation.pdf")
    run.log_artifact(artifact)
    run.finish()
    os.system("rm heatmap.pdf add_ablation.pdf remove_ablation.pdf")

for artifact_name in tqdm(decoder_ablation_artifacts):
    run = wandb.init(project="plmbias", name=f"{artifact_name.split(':')[0]}_visualizations")
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()
    with open(os.path.join(artifact_dir, "results.json"), "r") as file:
        results = json.loads(file.read())
    with open(os.path.join(artifact_dir, "shapes.json"), "r") as file:
        shapes = json.loads(file.read())

    contribs = results["contribs"]
    generate_heatmap(contribs, shapes["decoder"])

    generate_figure(results)

    artifact = wandb.Artifact(
        name=f"{artifact.name.split(':')[0]}_visualization",
        type="ablation_visualization"
    )
    artifact.add_file("heatmap.pdf")
    artifact.add_file("add_ablation.pdf")
    artifact.add_file("remove_ablation.pdf")
    run.log_artifact(artifact)
    run.finish()
    os.system("rm heatmap.pdf add_ablation.pdf remove_ablation.pdf")


