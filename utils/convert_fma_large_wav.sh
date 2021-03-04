#!/bin/bash
mkdir fma_large_8k 
for i in fma_large/**/*.mp3
do
    mkdir -p fma_large_8k/$(basename $(dirname "$i" .wav))
    sox --ignore-length "$i" "-r 8000" "-c 1" "-b 16" "fma_large_8k/$(basename $(dirname "$i" .wav))/$(basename "$i" .wav).wav"
done
