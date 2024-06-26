#!/bin/bash

cd llama_new

./llama-server \
    -m ./models/7B/ggml-model-q4_0.bin \
    --host "127.0.0.1" \
    --port 8080 \
    -c 4096 \
    -ngl 128 \
    --api-key "echo in the moon"

