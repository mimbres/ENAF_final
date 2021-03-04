#!/bin/bash
mkdir utube_reactoin_dataset_8k 
for i in utube_reaction_dataset/*.wav
do
    mkdir -p utube_reactoin_dataset_8k/$(basename $(dirname "$i" .wav))
    sox --ignore-length "$i" "-r 8000" "-c 1" "-b 16" "utube_reactoin_dataset_8k/$(basename $(dirname "$i" .wav))/$(basename "$i" .wav).wav"
done
