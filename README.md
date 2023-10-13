# WWW24-ID259-Span-Pair Interaction and Tagging for Dialogue-Level Aspect-Based Sentiment Quadruple Analysis

## Environment
- ``conda create -n mrm python==3.8``

- ``conda activate mrm``

- ``pip install -r requirements.txt``

## Data

We conduct experiments on Chinese and English datasets.
The datasets are located in `data/diaasq`.

## Running

1. training a Span Ranker.

- `cd span_ranker/src`
- `sh run_train.sh`

2. generating span indices and selecting the top K span by the trained span ranker.

- `cd span_ranker/src`
- `sh run_eval.sh`

3. training a MRM and obtain the results.

- `cd mrm/src`
- `sh run.sh`
