import wandb
from tqdm import tqdm
import pandas as pd
import os

api = wandb.Api()

results = pd.read_csv("results.csv")
os.environ["WANDB_SILENT"] = "true"

encoder_ablation_vis_artifacts = results[["name", "encoder_ablation_vis_artifact"]].dropna()
decoder_ablation_vis_artifacts = results[["name","decoder_ablation_vis_artifact"]].dropna()

for idx in tqdm(range(len(encoder_ablation_vis_artifacts))):
    artifact_name = encoder_ablation_vis_artifacts.iloc[idx]["encoder_ablation_vis_artifact"]
    output_name = encoder_ablation_vis_artifacts.iloc[idx]["name"]
    artifact = api.artifact(f"plmbias/{artifact_name}")
    artifact_dir = artifact.download()
    os.system(f"mkdir -p ./visualizations/results_emnlp/ablation_results/{output_name}/encoder")
    os.system(f"mv {artifact_dir}/* ./visualizations/results_emnlp/ablation_results/{output_name}/encoder")
    os.system(f"rm -r {artifact_dir}")

for idx in tqdm(range(len(decoder_ablation_vis_artifacts))):
    artifact_name = decoder_ablation_vis_artifacts.iloc[idx]["decoder_ablation_vis_artifact"]
    output_name = decoder_ablation_vis_artifacts.iloc[idx]["name"]
    artifact = api.artifact(f"plmbias/{artifact_name}")
    artifact_dir = artifact.download()
    os.system(f"mkdir -p ./visualizations/results_emnlp/ablation_results/{output_name}/decoder")
    os.system(f"mv {artifact_dir}/* ./visualizations/results_emnlp/ablation_results/{output_name}/decoder")
    os.system(f"rm -r {artifact_dir}")


