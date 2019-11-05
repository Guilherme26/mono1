#!/bin/bash

for epochs in 25 50 100; do
    for embedding in 64 128 256; do
        for n_layers in 1 2 3; do
            python3 main.py --n_hidden_units ${embedding} --train_epochs ${epochs} --n_hidden_layers ${n_layers} --relevance_threshold 50
        done
    done
done