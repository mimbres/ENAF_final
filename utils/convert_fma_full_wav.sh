#!/bin/bash
mkdir fma_full_8k 
for i in fma_full/**/*.mp3
do
    mkdir -p fma_full_8k/$(basename $(dirname "$i" .wav))
    sox --ignore-length "$i" "-r 8000" "-c 1" "-b 16" "fma_full_8k/$(basename $(dirname "$i" .wav))/$(basename "$i" .wav).wav"
done
