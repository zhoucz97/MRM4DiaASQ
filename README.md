# WWW24-ID259-Span-Pair Interaction and Tagging for Dialogue-Level Aspect-Based Sentiment Quadruple Analysis

## Environment
- ``conda create -n mrm python==3.8``

- ``conda activate mrm``

- [CUDA 11.1] ``pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html``

- ``pip install -r requirements.txt``

## Data

We conduct experiments on Chinese and English datasets.
The datasets are located in `data/diaasq`.

## Running

1. training a Span Ranker and saving the checkpoints to `data/saves/ranker`.

- `cd span_ranker/src`
- `bash run_train.sh`

2. generating span indices and selecting the top K span by the trained span ranker, saving the .pkl files to `data/saves/ranker`.

- `cd span_ranker/src`
- `bash run_eval.sh`

3. training a MRM and obtain the results.

- `cd mrm/src`
- `bash run.sh`
