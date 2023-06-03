from collections import defaultdict
import json
import os
import sys
import time

import docker


def get_docker_contexts(contexts):
    return {context: docker.DockerClient(base_url=url) for context, url in contexts.items()}


def launch_experiments(experiments, context_urls):
    wandb_token = os.environ.get("WANDB_TOKEN")
    for experiment in experiments:
        buildargs = {k: str(v) for k, v in experiment["buildargs"].items()}
        buildargs["CUDA_VISIBLE_DEVICES"] = str(experiment["card"])
        buildargs["WANDB_API_KEY"] = wandb_token
        print("Launching container...")
        env_str = " ".join([f"-e {key}={value}" for key, value in buildargs.items()])
        os.system(f"docker context use {experiment['context']} && docker run {env_str} -e WANDB_DOCKER={experiment['image']} -itd --gpus all --name {experiment['name']} {experiment['image']}")
        print(f"Started Experiment: {experiment['name']}")
        time.sleep(1)



def monitor_experiments(experiments, context_urls):
    contexts = get_docker_contexts(context_urls)
    print(f"\033[94m \033[1m{'Name':<50} \033[0m{'Machine':<12}  {'Card':<5} {'Status':<10} | {'Logs'}")
    for experiment in experiments:
        client = contexts[experiment["context"]]
        try:
            container = client.containers.get(experiment["name"])
            print(f"\033[94m \033[1m{experiment['name']:<50} \033[0m{experiment['context']:<12}  {experiment['card']:<5} {container.status:<10} | {str(container.logs(tail=1))[:100]}")
        except docker.errors.NotFound:
            print(f"Container \"{experiment['name']}\" does not exist")


def stop_experiments(experiments, context_urls):
    context_to_expers = defaultdict(list)

    for experiment in experiments:
        context_to_expers[experiment["context"]] += [experiment['name']]
    
    for context in context_to_expers.keys():
        os.system(f"docker context use {context} && docker rm --force {' '.join([name for name in context_to_expers[context]])}")



if __name__ == "__main__":
    with open(sys.argv[2]) as file:
        argStr = "".join(file.readlines())
    obj = json.loads(argStr)
    prebuild = obj["prebuild"] if "prebuild" in obj.keys() else None
    experiments = obj["experiments"]
    contexts = obj["contexts"]
    if sys.argv[1] == "launch":
        if prebuild is not None:
            build_images(prebuild, contexts)
        launch_experiments(experiments, contexts)
    elif sys.argv[1] == "monitor":
        monitor_experiments(experiments, contexts)
    elif sys.argv[1] == "stop":
        stop_experiments(experiments, contexts)
    else:
        print("Invalid Command")
