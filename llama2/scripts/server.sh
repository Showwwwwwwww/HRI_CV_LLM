#!/bin/bash

cd llama_new

CUDA_VISIBLE_DEVICES=1,2 ./llama-server \
    -m ./models/8B/Llama-3-8B-Instruct-Gradient-1048k-Q6_K.gguf \
    --host "127.0.0.1" \
    --port 8080 \
    -c 4096 \
    -ngl 256 \
    --api-key "echo in the moon"

