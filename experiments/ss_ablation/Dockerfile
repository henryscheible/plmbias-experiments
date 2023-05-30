FROM henryscheible/plmbias@sha256:c5e00ff748fb74ab6215cc704a04aa53a038c7dd33569fe58dab60177e6fc563
ARG TOKEN
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
ARG CHECKPOINT
ARG GPU_CARD
ENV CHECKPOINT=$CHECKPOINT
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV TOKENIZERS_PARALLELISM=false
COPY ./train.py /workspace
CMD ["python3", "/workspace/train.py"]
