#!/usr/bin/env bash

(git clone https://github.com/NVIDIA/apex && wget git https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz) &
conda create -n fairseq python=3
source activate fairseq
conda install pytorch -c pytorch
wait
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ..
