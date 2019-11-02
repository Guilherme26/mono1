#!/bin/bash

for i in 64 128 256; do
    python3 main.py --n_hidden_units ${i} --train_epochs 300 --relevance_threshold 35
done