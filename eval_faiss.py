#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 08:00:01 2019

Large-scale evaluation with saved embeddings

USAGE:
    python eval_faiss.py [argv1: CPU or GPU] [argv2: EMBEDDING_DIR] [argv3: N_SAMPLE_TEST] [argv4: SEARCH_MODE] [argv5: N_PROBE]
    
    - argv1: (str) <Required> Embedding directory
    - argv2: (int) number of test samples. (default=2000)
    - argv2: (str) <optional> 'l2': (default)
                              'ivf':
                              'pq':
                              'ivfpq':
                              'ivfpq-rr':
                              'ivfpq-rr-ondisk':
                              'hnsw':
    - argv3: (int) <optional> Number of probe, (default=100)

------------------------------------------------------------------------------
2020 results with NTxent loss + MIREX
   
<MIREX +10k(30s)>
(final-1280-aug1-speech-1010db)
eval_faiss.py gpu logs/emb/final_exp_NTxent_simCLR_1280win1_specaug1_sp-1010db/100/10k 2000 ivfpq-rr 20 mirex
matched_exact=[ 8.67 14.33 16.49 18.76 19.79 21.11]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[33.55 49.95 59.57 72.67 76.34 82.66]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[35.53 52.03 60.98 73.99 77.95 83.79]%, scope=[1, 3, 5, 9, 11, 19]

(final-640-aug1-speech-1010db-LAMB)
matched_exact=[ 9.9  14.89 16.68 19.04 19.23 21.02]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[36.48 51.65 60.79 73.33 75.4  81.81]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[36.48 51.65 60.79 73.33 75.4  81.81]%, scope=[1, 3, 5, 9, 11, 19]

(final-640-aug1-speech-1010db)
eval_faiss.py gpu logs/emb/final_exp_NTxent_simCLR_640win1_specaug1_sp-1010db/100/10k 2000 ivfpq-rr 20 mirex
matched_exact=[ 8.95 13.85 16.49 19.23 19.6  21.39]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[33.55 49.39 59.94 73.42 76.15 82.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[35.91 51.56 61.73 74.65 77.76 83.03]%, scope=[1, 3, 5, 9, 11, 19]


(640-aug05)
matched_exact=[ 7.06 12.37 14.79 17.32 18.01 19.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[30.83 47.33 55.72 64.17 66.81 73.34]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[32.15 48.34 56.57 64.87 67.53 74.02]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug1/100/10k 2000 ivfpq-rr 20 mirex
matched_exact=[ 7.11 12.51 14.9  17.25 17.88 19.98]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[30.87 47.67 56.18 65.3  67.94 75.08]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[32.3  48.7  57.09 66.02 68.66 75.83]%, scope=[1, 3, 5, 9, 11, 19]

matched_exact=[ 7.07 12.06 14.23 16.21 17.06 18.47]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[27.71 42.51 52.69 63.43 67.86 72.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[29.22 43.64 53.44 64.   68.43 73.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-3h2s)
matched_exact=[ 5.46  9.6  11.66 13.95 14.7  16.83]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[25.52 38.93 46.73 54.99 57.58 64.54]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[27.18 40.05 47.54 55.69 58.26 65.22]%, scope=[1, 3, 5, 9, 11, 19]

(640 coordconv)
matched_exact=[ 6.6  11.03 13.1  14.99 15.93 17.62]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[26.48 41.09 49.01 61.36 64.75 69.27]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[27.9  42.51 50.05 61.83 65.13 69.84]%, scope=[1, 3, 5, 9, 11, 19]

(320 coordconv Large front param)
matched_exact=[ 5.75 11.22 13.57 15.46 15.83 17.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[25.16 41.94 51.56 61.64 64.28 68.61]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[26.48 43.17 52.5  62.11 64.66 69.18]%, scope=[1, 3, 5, 9, 11, 19]


<MIREX +100k(30s)>
(640-aug1)
matched_exact=[ 5.    8.58 10.56 12.82 13.29 15.36]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[20.45 34.02 42.13 52.78 56.36 63.34]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[21.21 35.06 42.98 53.44 57.21 64.18]%, scope=[1, 3, 5, 9, 11, 19]

------------------------------------------------------------------------------
2020 results with NTxent loss
<10k(30s)>
(final-640-aug1-speech-1010db-LAMB)
matched_exact=[84.05 96.3  97.7  98.65 99.05 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.05 96.55 97.8  98.85 99.15 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.05 96.55 97.8  98.85 99.15 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(final-640-aug1-speech-1010db)
eval_faiss.py gpu logs/emb/final_exp_NTxent_simCLR_640win1_specaug1_sp-1010db/100/10k 2000 ivfpq-rr 20 
matched_exact=[79.41 95.28 97.79 98.79 99.25 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.38 95.73 97.84 98.9  99.3  99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.06 97.94 99.   99.45 99.55 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(120)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120_tau0p05_ln_adam_ddrop_lr5e-5/45/10k 2000 ivfpq-rr 20
matched_exact=[80.05 94.45 97.   98.2  98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.55 95.9  97.6  98.5  99.   99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.2  98.2  98.8  99.25 99.45 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(120-trainable-tau)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_120_specaug_trtau/100/10k 2000 ivfpq-rr 20
matched_exact=[77.2  93.7  96.8  98.2  98.65 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.75 95.4  97.6  98.4  98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.3 98.  99.  99.2 99.3 99.5]%, scope=[1, 3, 5, 9, 11, 19]

(120-fixtau001)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_120_fixtau001/100/10k 2000 ivfpq-rr 20
matched_exact=[75.65 94.2  96.7  98.05 98.75 99.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.8  95.7  97.2  98.35 98.85 99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.5  98.2  98.65 99.15 99.35 99.45]%, scope=[1, 3, 5, 9, 11, 19]

(320)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_320_ln_lamb_lr13e-5/24/10k 2000 ivfpq-rr 20
matched_exact=[80.8  94.95 97.4  98.7  99.05 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.8  96.25 97.8  98.85 99.2  99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.95 98.4  98.95 99.4  99.55 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320-noBN-fc-tau003)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau003/20/10k 2000 ivfpq-rr 20
matched_exact=[77.4  93.75 96.55 98.25 98.75 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.   95.7  97.3  98.55 98.95 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.8  97.9  98.9  99.25 99.35 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau003/52/10k 2000 ivfpq-rr 20
matched_exact=[81.6  94.85 96.9  98.4  98.95 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.75 96.15 97.35 98.6  99.05 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.25 98.3  98.9  99.3  99.4  99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau003/99/10k 2000 ivfpq-rr 20
matched_exact=[79.7  94.   96.75 98.45 98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.4  95.7  97.05 98.5  99.   99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.55 98.35 98.85 99.35 99.45 99.75]%, scope=[1, 3, 5, 9, 11, 19]

(320-noBN-fc-tau005)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau005/20/10k 2000 ivfpq-rr 20
matched_exact=[81.3  94.65 97.25 98.6  99.   99.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.4 95.9 97.9 98.8 99.1 99.7]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.1  98.15 98.95 99.45 99.6  99.75]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau005/52/10k 2000 ivfpq-rr 20
matched_exact=[80.3  93.85 96.9  98.25 98.75 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.15 95.2  97.3  98.45 98.85 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.95 97.65 98.75 99.15 99.35 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau005/100/10k 2000 ivfpq-rr 20
matched_exact=[75.85 93.15 96.65 98.35 98.7  99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.4  94.2  97.15 98.45 98.8  99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.   97.3  98.5  98.95 99.05 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320-noBN-fc-tau01)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau01/20/10k 2000 ivfpq-rr 20
matched_exact=[79.5  94.1  96.65 98.2  98.8  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.4  95.85 97.4  98.4  98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.9  98.2  98.95 99.1  99.45 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau01/52/10k 2000 ivfpq-rr 20
matched_exact=[73.05 93.15 95.3  97.4  98.2  99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.75 94.75 96.15 97.9  98.65 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[90.1  97.25 98.2  98.85 99.25 99.55]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_noBN_tau01/99/10k 2000 ivfpq-rr 20
matched_exact=[59.   85.95 93.15 96.25 97.   98.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[69.05 88.95 94.45 96.9  97.55 98.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[79.45 93.95 96.95 98.15 98.55 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]


(320-divenc-tau001)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau001/21/10k 2000 ivfpq-rr 20
matched_exact=[75.55 93.4  96.   98.15 98.7  99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.85 94.8  96.6  98.4  98.85 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[90.25 97.35 98.15 99.2  99.4  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau001/51/10k 2000 ivfpq-rr 20
matched_exact=[76.35 93.65 96.1  98.1  98.5  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.8  94.95 96.55 98.4  98.7  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.35 97.7  98.35 99.1  99.2  99.45]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau001/99/10k 2000 ivfpq-rr 20
matched_exact=[76.95 93.7  96.4  98.4  98.85 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.15 94.85 96.9  98.7  98.95 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.15 97.65 98.9  99.4  99.45 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-divenc-tau002)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau002/21/10k 2000 ivfpq-rr 20
matched_exact=[77.65 93.8  96.55 98.15 98.65 99.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.55 95.2  97.1  98.4  98.95 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[90.95 97.5  98.5  99.1  99.3  99.45]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau002/51/10k 2000 ivfpq-rr 20
matched_exact=[78.65 93.85 96.55 98.55 98.8  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.2 95.6 97.3 98.6 98.9 99.4]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.4  98.   98.75 99.1  99.25 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau002/174/10k 2000 ivfpq-rr 20
matched_exact=[80.9  95.1  97.1  98.5  98.75 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.7  95.95 97.6  98.7  98.9  99.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.05 98.05 98.85 99.35 99.4  99.75]%, scope=[1, 3, 5, 9, 11, 19]

(320-divenc-tau003)
python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_tau005/21/10k 2000 ivfpq-rr 20
matched_exact=[80.15 95.05 97.3  98.65 98.75 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.55 96.05 97.55 98.7  98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.95 98.1  98.7  99.4  99.4  99.65]%, scope=[1, 3, 5, 9, 11, 19]

51
matched_exact=[82.05 95.25 97.35 98.7  99.   99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.35 96.6  97.85 98.85 99.1  99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.1  98.35 99.15 99.45 99.65 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-divenc-tau005)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau005/21/10k 2000 ivfpq-rr 20
matched_exact=[81.45 94.75 97.05 98.55 98.85 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.6  95.95 97.55 98.7  99.05 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.35 98.15 98.95 99.35 99.35 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau005/51/10k 2000 ivfpq-rr 20
matched_exact=[79.6  94.35 96.65 98.3  98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.9  95.55 97.2  98.4  98.95 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.85 97.9  98.6  99.3  99.35 99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320win1_server_fc_divenc_tau005/99/10k 2000 ivfpq-rr 20
matched_exact=[76.85 92.95 96.2  98.3  98.55 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.35 94.65 97.   98.55 98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.8  97.35 98.25 99.05 99.2  99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320-lars-tau005)
python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_tau005/21/10k 2000 ivfpq-rr 20
matched_exact=[75.95 93.4  96.35 98.1  98.45 99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.75 95.2  97.05 98.35 98.85 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.2  97.7  98.7  99.   99.35 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_tau005/51/10k 2000 ivfpq-rr 20
matched_exact=[78.25 94.05 96.6  98.15 98.55 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.4  95.5  97.25 98.3  98.9  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.95 97.85 98.75 99.1  99.35 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_tau005/99/10k 2000 ivfpq-rr 20
matched_exact=[78.5  93.8  96.8  98.05 98.55 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.5  95.5  97.6  98.5  98.85 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.3  97.65 98.85 99.35 99.35 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320-lars-tau01 paper)
python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_paper/21/10k 2000 ivfpq-rr 20
matched_exact=[81.8  94.7  97.05 98.2  98.75 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.3  96.25 97.5  98.35 98.85 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.5  98.2  98.95 99.2  99.35 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_paper/51/10k 2000 ivfpq-rr 20
matched_exact=[79.8  94.35 96.8  98.5  98.95 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.9  96.1  97.45 98.6  99.   99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.75 97.85 98.7  99.2  99.35 99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb//exp_NTxent_simCLR_320win1_server_divenc_lars_paper/99/10k 2000 ivfpq-rr 20
matched_exact=[79.   94.15 96.85 98.25 98.6  99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.8  95.6  97.35 98.35 98.75 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.2  97.95 98.85 99.2  99.35 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_specaug05/62/10k 2000 ivfpq-rr 20
matched_exact=[84.1  95.1  97.1  98.5  98.85 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.55 96.45 97.8  98.75 99.   99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.8  98.35 98.9  99.4  99.45 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_specaug1/71/10k 2000 ivfpq-rr 20
matched_exact=[83.7  95.85 97.5  98.75 99.   99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.95 96.75 97.95 98.9  99.15 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.35 98.55 99.15 99.45 99.6  99.65]%, scope=[1, 3, 5, 9, 11, 19]

(640)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_lr1e04/59/10k 2000 ivfpq-rr 20
matched_exact=[82.   94.75 97.1  98.6  98.95 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.5  96.35 97.45 98.65 99.05 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.75 98.   98.7  99.4  99.45 99.75]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05/61/10k 2000 ivfpq-rr 20
matched_exact=[83.55 95.2  97.2  98.95 98.9  99.55]%, scope=[1, 3, 5, 9, 11, 19]                           
matched_near=[90.8  96.3  97.85 99.05 99.05 99.55]%, scope=[1, 3, 5, 9, 11, 19]                            
matched_song=[95.3  98.15 98.95 99.45 99.5  99.7 ]%, scope=[1, 3, 5, 9, 11, 19] 

(640-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug1/100/10k 2000 ivfpq-rr 20
matched_exact=[83.55 95.4  97.35 98.75 98.95 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.85 96.55 97.7  98.8  99.05 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.65 98.5  99.   99.4  99.5  99.8 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05-d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05_d64/100/10k 2000 ivfpq-rr 20
matched_exact=[81.65 94.7  97.05 98.75 99.05 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.4  95.85 97.3  98.85 99.15 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.85 98.   98.75 99.3  99.45 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug08)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug08_3c2s/295/10k 2000 ivfpq-rr 20
matched_exact=[81.1  93.85 96.2  98.2  98.8  99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.95 95.8  97.15 98.5  99.1  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.2  97.9  98.6  99.25 99.45 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug065)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug065_3c2s/339/10k 2000 ivfpq-rr 2
matched_exact=[81.   93.95 96.5  98.25 98.9  99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.9  96.   97.35 98.55 99.2  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.2  98.   98.8  99.25 99.45 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(1280-noaug)
python eval_faiss.py cpu logs/emb/1280win1_nospecaug/100/10k 2000 ivfpq-rr 20
matched_exact=[76.1  91.95 95.75 97.5  98.15 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.9  93.8  96.45 97.9  98.25 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.85 96.7  98.1  98.7  98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau001)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau001/21/10k 2000 ivfpq-rr 20
matched_exact=[78.5  93.85 96.3  98.2  98.4  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.4  94.8  96.95 98.3  98.55 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[90.55 97.2  98.35 98.95 99.05 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau001/51/10k 2000 ivfpq-rr 20
matched_exact=[78.35 94.2  96.7  98.15 98.7  99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.1  95.05 97.15 98.15 98.75 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[90.05 96.95 98.35 98.7  99.   99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau001/99/10k 2000 ivfpq-rr 20
matched_exact=[78.45 94.2  96.9  98.7  98.85 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.35 95.   97.25 98.75 98.95 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.45 97.4  98.55 99.15 99.3  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau002)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau002/21/10k 2000 ivfpq-rr 20
matched_exact=[78.5  93.9  96.1  98.6  98.75 99.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[84.85 95.45 96.7  98.7  98.9  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.65 97.55 98.25 99.05 99.2  99.55]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau002/51/10k 2000 ivfpq-rr 20
matched_exact=[79.55 94.3  96.55 98.25 98.65 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.6  95.2  97.   98.4  98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.8  97.45 98.4  99.   99.15 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau002/51/10k 2000 ivfpq-rr 20
matched_exact=[80.25 94.5  96.7  98.45 98.75 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.9  95.6  97.2  98.5  98.85 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.5  97.65 98.65 99.   99.25 99.55]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau003)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau003/21/10k 2000 ivfpq-rr 20
matched_exact=[79.6  95.15 96.95 98.6  98.6  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.95 96.35 97.45 98.65 98.75 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.1  98.2  98.65 99.15 99.25 99.55]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau003/51/10k 2000 ivfpq-rr 20
matched_exact=[81.1  94.55 97.05 98.55 98.8  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.9  95.65 97.7  98.7  98.9  99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.45 97.8  98.95 99.25 99.3  99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau003/99/10k 2000 ivfpq-rr 20
matched_exact=[81.8  94.75 96.8  98.55 98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.35 96.   97.5  98.65 99.05 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.75 98.35 98.9  99.3  99.45 99.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau005)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau005/21/10k 2000 ivfpq-rr 20
matched_exact=[80.5  95.05 96.95 98.4  99.   99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.75 96.05 97.5  98.6  99.15 99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.65 98.1  98.7  99.2  99.45 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau005/51/10k 2000 ivfpq-rr 20
matched_exact=[79.35 94.45 96.85 98.65 98.9  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.5  95.9  97.45 98.85 99.   99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.2  97.95 98.65 99.2  99.3  99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau005/99/10k 2000 ivfpq-rr 20
matched_exact=[78.1  93.85 96.2  98.55 98.7  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[86.15 95.25 96.75 98.7  98.9  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.55 97.4  98.3  99.15 99.3  99.65]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau005-wd10e6): large weight decay is bad
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau005w_d10e6/21/10k 2000 ivfpq-rr 20
matched_exact=[70.25 90.75 94.2  96.8  97.5  98.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[76.8  92.55 94.95 97.   97.7  98.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.25 94.95 96.75 97.8  98.25 98.6 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau01)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01/21/10k 2000 ivfpq-rr 20
matched_exact=[79.05 94.15 96.4  98.2  98.7  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.4  95.3  97.05 98.4  98.8  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.05 97.25 98.3  99.   99.1  99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01/51/10k 2000 ivfpq-rr 20
matched_exact=[74.  92.  95.5 97.5 98.1 99.1]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.65 93.85 96.15 97.8  98.25 99.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[89.75 96.2  97.65 98.7  98.9  99.45]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01/91/10k 2000 ivfpq-rr 20
matched_exact=[62.85 87.05 93.55 96.5  97.6  98.8 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[71.85 90.05 94.5  97.05 98.   99.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[81.   93.95 96.8  98.35 98.65 99.3 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-clip3-divenc-noaug-tau01-wd10e6): large weight decay is bad
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01_wd10e6/20/10k 2000 ivfpq-rr 20
matched_exact=[78.15 93.65 96.65 98.15 98.65 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[87.85 95.15 97.25 98.35 98.8  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[93.25 97.25 98.5  99.   99.15 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01_wd10e6/50/10k 2000 ivfpq-rr 20
matched_exact=[74.55 92.05 95.6  97.4  98.   99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[83.1  93.7  96.15 97.6  98.15 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[89.85 96.25 97.95 98.65 98.8  99.55]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_1280_noaug_TPU_clip3_lr02_divenc_tau01_wd10e6/99/10k 2000 ivfpq-rr 20
matched_exact=[61.4  86.9  92.5  96.5  97.5  98.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[70.7  89.6  93.7  97.05 97.8  98.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[80.4  93.55 96.45 98.45 98.75 99.35]%, scope=[1, 3, 5, 9, 11, 19]

(640 coordconv)
python eval_faiss.py cpu logs/emb/NTxent_simCLR_640win1_specaug1_coordconv/100/10k 2000 ivfpq-rr 20
matched_exact=[84.1  95.2  97.05 98.85 99.   99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.75 96.7  97.6  99.   99.1  99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.55 98.25 98.95 99.45 99.45 99.55]%, scope=[1, 3, 5, 9, 11, 19]

(320 coordconv Large front param)
python eval_faiss.py cpu logs/emb/NTxent_simCLR_320win1_specaug1_coordconvL/96/10k 2000 ivfpq-rr 20
matched_exact=[82.75 95.2  97.35 98.45 98.95 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.55 96.65 97.9  98.55 99.1  99.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.45 98.4  99.1  99.3  99.45 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320 coordconv 2dconv)
matched_exact=[83.1  95.   97.35 98.85 98.85 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[89.85 96.35 97.8  98.9  99.   99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[94.75 98.25 98.95 99.35 99.45 99.65]%, scope=[1, 3, 5, 9, 11, 19]

(320 2dconv)
matched_exact=[83.5  95.   97.   98.45 98.8  99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[90.4  96.5  97.45 98.65 98.95 99.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[95.25 98.35 98.85 99.4  99.45 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]

(old)
python eval_faiss.py cpu logs/emb/exp_v2fix_semihard_320win1_d64/20/10k 2000 ivfpq-rr 20
matched_exact=[58.55 86.65 92.15 95.65 96.3  97.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[68.25 89.   92.85 95.9  96.7  97.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[75.6  92.15 94.8  96.7  97.25 97.9 ]%, scope=[1, 3, 5, 9, 11, 19]


<100k(30s)>
(120)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120_tau0p05_ln_adam_ddrop_lr5e-5/45/100k 2000 ivfpq-rr 20
matched_exact=[70.8  89.35 91.9  95.1  95.7  97.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[77.35 90.7  92.55 95.3  96.   97.8 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[83.75 93.8  94.85 96.4  96.7  98.1 ]%, scope=[1, 3, 5, 9, 11, 19]

(120-trainable-tau)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_120_specaug_trtau/100/100k 2000 ivfpq-rr 20
matched_exact=[69.15 89.25 92.6  95.45 96.   97.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[77.5  90.5  93.4  95.75 96.2  97.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.05 94.05 95.65 96.55 96.75 97.9 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_specaug05/62/100k 2000 ivfpq-rr 20
matched_exact=[75.3  91.9  94.15 96.45 96.65 98.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.2  92.7  94.5  96.55 96.75 98.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.3  95.2  96.1  97.25 97.4  98.3 ]%, scope=[1, 3, 5, 9, 11, 19]

(320-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_specaug1/71/100k 2000 ivfpq-rr 20
matched_exact=[75.8 91.1 93.7 96.5 96.8 98. ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.65 92.   94.2  96.55 96.95 98.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.35 94.95 96.2  97.4  97.4  98.25]%, scope=[1, 3, 5, 9, 11, 19]

(640)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_lr1e04/59/100k 2000 ivfpq-rr 20
matched_exact=[74.25 90.1  93.1  95.7  96.1  97.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.45 91.15 93.5  95.95 96.3  97.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.3  94.75 95.8  96.8  96.9  98.2 ]%, scope=[1, 3, 5, 9, 11, 19]

(640) faiss codesize 128 re
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_lr1e04/59/100k 2000 ivfpq-rr128 20
matched_exact=[74.25 90.25 93.1  95.7  96.15 97.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.45 91.35 93.5  96.   96.4  97.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.3  94.85 95.8  96.85 97.   98.2 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug1/100/100k 2000 ivfpq-rr 20
matched_exact=[75.7  90.85 94.05 96.3  96.8  98.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.6  91.95 94.4  96.45 96.95 98.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.65 94.8  96.15 97.15 97.35 98.2 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05/61/10k 2000 ivfpq-rr 20
atched_exact=[75.95 91.85 93.95 96.35 96.85 98.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[83.2  92.85 94.25 96.4  96.9  98.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[89.05 95.35 96.05 97.15 97.4  98.25]%, scope=[1, 3, 5, 9, 11, 19]

(640d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_d64_ln_drop_lamb/70/100k 2000 ivfpq-rr 20
matched_exact=[72.1  89.4  92.35 95.25 95.95 97.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[78.6  90.75 92.9  95.4  96.1  97.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.5  94.1  95.   96.7  96.9  97.85]%, scope=[1, 3, 5, 9, 11, 19]

(640d32)??
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_d32/12/100k 2000 ivfpq-rr32 20
matched_exact=[54.45 82.   87.8  92.25 93.3  96.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[61.3  84.45 88.95 92.85 93.65 96.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[68.6  88.9  92.15 94.85 95.15 97.4 ]%, scope=[1, 3, 5, 9, 11, 19]

(640d16)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_d16/50/100k 2000 ivfpq-rr16 20
matched_exact=[25.6  65.3  76.95 86.8  88.7  93.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[30.95 68.   78.6  87.3  89.35 94.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[36.9  72.95 82.9  89.4  90.95 95.6 ]%, scope=[1, 3, 5, 9, 11, 19]

(120d16)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120_d16/15/100k 2000 ivfpq-rr16 20
matched_exact=[30.2  71.7  81.15 88.55 90.45 94.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[34.75 74.7  83.1  89.45 91.1  95.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[39.95 79.6  86.95 91.4  92.6  96.2 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05-d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05_d64/100/100k 2000 ivfpq-rr 20
matched_exact=[71.55 89.55 92.8  95.35 95.65 97.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[78.65 90.9  93.1  95.5  95.8  97.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.8  93.9  95.35 96.7  96.95 97.85]%, scope=[1, 3, 5, 9, 11, 19]

(seen / 640d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_d64_ln_drop_lamb_seen/112/100k 2000 ivfpq-rr 20
matched_exact=[74.65 90.6  93.2  96.25 97.   98.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[82.9  91.65 93.8  96.4  97.15 98.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.6  95.1  96.05 97.45 97.6  98.4 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug065)
matched_exact=[72.55 88.8  92.   95.75 95.95 97.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[80.7  90.35 92.85 95.95 96.1  97.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.4  94.3  95.8  97.05 97.   97.9 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug08)
matched_exact=[72.3  89.35 93.25 96.1  96.35 97.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.2  91.35 94.   96.15 96.45 97.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.9  94.75 96.35 97.25 97.3  98.2 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-noaug)
python eval_faiss.py cpu logs/emb/1280win1_nospecaug/100/100k 2000 ivfpq-rr 20
matched_exact=[65.4  86.2  91.1  94.   95.   96.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[73.6  88.15 91.8  94.35 95.2  96.8 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[80.85 91.95 94.25 95.6  96.   97.05]%, scope=[1, 3, 5, 9, 11, 19]

(640 coordconv)
python eval_faiss.py cpu logs/emb/ENAF_coordconv/NTxent_simCLR_640win1_specaug1_coordconv/100/100k 2000 ivfpq-rr 20
matched_exact=[75.25 90.7  93.3  96.05 96.6  97.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.8  91.55 93.7  96.2  96.75 97.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.15 95.15 96.1  97.   97.3  98.15]%, scope=[1, 3, 5, 9, 11, 19]

(320 coordconv Large front param)
python eval_faiss.py cpu logs/emb/ENAF_coordconv/NTxent_simCLR_320win1_specaug1_coordconvL/96/100k 2000 ivfpq-rr 20
matched_exact=[74.65 90.6  93.3  96.   96.5  97.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.9  91.45 93.65 96.2  96.6  97.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.85 94.65 95.8  97.1  97.3  98.  ]%, scope=[1, 3, 5, 9, 11, 19]

(320 coordconv 2dconv)
python eval_faiss.py cpu logs/emb/NTxent_simCLR_320win1_specaug1_coordconv2d/100/100k 2000 ivfpq-rr 20
matched_exact=[74.7 90.8 93.7 95.8 96.5 98. ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[80.9  91.75 94.3  96.15 96.7  98.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[87.6  94.9  96.15 96.95 97.25 98.15]%, scope=[1, 3, 5, 9, 11, 19]

(320 2dconv)
python eval_faiss.py cpu logs/emb/NTxent_simCLR_320win1_specaug1_conv2d/100/100k 2000 ivfpq-rr 20
matched_exact=[75.   90.   93.15 96.1  96.65 98.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.05 91.2  93.5  96.35 96.85 98.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[86.75 94.5  95.7  97.1  97.3  98.1 ]%, scope=[1, 3, 5, 9, 11, 19]

(old)
python eval_faiss.py cpu logs/emb/exp_v2fix_semihard_320win1_d64/20/100k 2000 ivfpq-rr 20
matched_exact=[44.2  75.05 84.05 89.3  91.2  94.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[51.9  77.2  85.4  89.75 91.5  94.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[58.9  82.65 88.75 91.9  93.05 95.4 ]%, scope=[1, 3, 5, 9, 11, 19]





<100k(full)>
(final-640-aug1-speech-1010db-LAMB)
matched_exact=[60.   81.65 87.6  92.   93.2  95.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[62.55 82.3  87.85 92.35 93.45 95.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[62.55 82.3  87.85 92.35 93.45 95.75]%, scope=[1, 3, 5, 9, 11, 19]

(final-640-aug1-speech-1010db)
matched_exact=[57.46 80.26 86.69 91.66 92.92 95.68]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[60.02 81.06 86.99 91.71 93.02 95.73]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[71.22 87.09 91.06 94.32 94.93 96.84]%, scope=[1, 3, 5, 9, 11, 19
                                                            
(120)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120_tau0p05_ln_adam_ddrop_lr5e-5/45/100k_full 2000 ivfpq-rr 20
matched_exact=[55.9  78.75 84.9  90.85 92.15 95.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[62.3  80.9  86.25 91.5  92.8  95.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[73.05 87.4  90.85 94.   94.7  96.1 ]%, scope=[1, 3, 5, 9, 11, 19]

(120-trainable-tau)
matched_exact=[51.3  77.6  83.75 90.1  91.3  93.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[58.35 79.85 85.2  91.05 91.85 94.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[70.6  86.65 90.35 93.4  93.95 95.45]%, scope=[1, 3, 5, 9, 11, 19]

(120-fixtau001)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_120_fixtau001/100/100k_full 2000 ivfpq-rr 20
matched_exact=[49.35 76.95 83.95 90.2  91.3  94.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[56.2  79.45 85.4  91.   92.   95.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[68.05 86.15 90.2  93.9  94.3  96.  ]%, scope=[1, 3, 5, 9, 11, 19]

(320)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_ln_drop_lamb_sch_cos_lr1e04/57/100k_full 2000 ivfpq-rr 20
matched_exact=[58.95 80.15 85.35 91.35 92.3  95.3 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[64.7  82.35 86.75 92.   92.95 95.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[76.2  88.75 91.1  94.1  94.8  96.55]%, scope=[1, 3, 5, 9, 11, 19]

(320-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_320_specaug05/62/100k_full 2000 ivfpq-rr 20
matched_exact=[61.   82.15 87.05 92.15 93.55 95.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[67.1  84.1  88.05 92.5  93.9  95.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[79.35 90.25 92.   94.85 95.55 96.65]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05/61/100k_full 2000 ivfpq-rr 20
matched_exact=[62.05 82.15 86.75 92.   93.55 95.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[67.5  83.5  87.85 92.5  94.05 95.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[78.7  89.5  92.3  94.75 95.65 96.85]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug1)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug1/100/100k_full 2000 ivfpq-rr 20
matched_exact=[62.2  83.2  87.35 92.   93.3  95.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[68.25 84.9  88.65 92.65 94.05 95.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[80.35 90.5  92.9  95.1  95.9  96.9 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug05-d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug05_d64/100/100k_full 2000 ivfpq-rr 20
matched_exact=[56.35 78.55 84.5  90.6  92.4  94.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[62.15 81.15 85.65 91.4  92.9  95.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[74.3  87.9  90.7  94.15 94.9  96.3 ]%, scope=[1, 3, 5, 9, 11, 19]

(640d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_d64_ln_drop_lamb/70/100k_full 2000 ivfpq-rr 20
matched_exact=[54.55 78.9  85.35 90.4  92.   94.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[61.25 81.65 86.65 90.9  92.65 95.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[72.35 87.2  90.55 93.6  94.5  96.15]%, scope=[1, 3, 5, 9, 11, 19]

(640d32)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_d32/50/100k_full 2000 ivfpq-rr32 20
matched_exact=[34.45 66.65 77.15 84.25 86.2  91.1 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[39.85 69.35 78.5  85.45 87.2  91.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[50.7  77.45 83.95 89.2  90.25 93.8 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_d32/32/100k_full 2000 ivfpq-rr32 20
matched_exact=[40.15 69.7  79.25 86.2  87.9  92.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[45.55 72.25 80.5  87.1  88.8  92.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[57.25 80.2  86.   90.55 91.75 94.  ]%, scope=[1, 3, 5, 9, 11, 19]

(640d16)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_d16/50/100k_full 2000 ivfpq-rr16 20
matched_exact=[11.6  45.2  57.65 69.1  72.55 80.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[14.55 47.3  59.75 71.   74.4  81.75]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[20.9  55.45 66.9  75.9  78.45 84.7 ]%, scope=[1, 3, 5, 9, 11, 19]

(640)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_ln_drop_lamb_lr1e04/59/100k_full 2000 ivfpq-rr 20
matched_exact=[59.4  81.7  86.8  91.25 92.75 95.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[66.35 83.75 87.75 92.05 93.3  95.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[77.6  90.   92.15 94.8  95.2  96.55]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug065)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug065_3c2s/339/100k_full 2000 ivfpq-rr 20
matched_exact=[58.6  78.95 84.2  90.1  91.5  94.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[66.85 81.55 86.3  91.1  92.15 94.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[78.6  88.6  91.4  93.65 94.45 95.9 ]%, scope=[1, 3, 5, 9, 11, 19]

(640-aug08)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_specaug08_3c2s/295/100k_full 2000 ivfpq-rr 2
matched_exact=[59.05 79.35 85.4  90.5  92.25 94.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[66.   82.15 87.15 91.65 92.9  94.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[78.75 89.2  91.75 94.55 94.9  96.3 ]%, scope=[1, 3, 5, 9, 11, 19]

(1280-naug)
python eval_faiss.py cpu logs/emb/1280win1_nospecaug/100/100k_full 2000 ivfpq-rr 20
matched_exact=[49.95 72.75 80.2  86.75 88.65 92.4 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[56.55 75.3  81.45 87.45 89.15 92.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[68.35 83.7  87.   90.8  91.6  93.95]%, scope=[1, 3, 5, 9, 11, 19]

(640 coordconv)
python eval_faiss.py cpu logs/emb/ENAF_coordconv/NTxent_simCLR_640win1_specaug1_coordconv/100/100k_full 2000 ivfpq-rr 20


(320 coordconv Large front param)
-not tested-


(old)
python eval_faiss.py cpu logs/emb/exp_v2fix_semihard_320win1_d64/20/100k_full 2000 ivfpq-rr 20
matched_exact=[25.75 58.45 69.25 78.45 81.4  87.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[30.85 61.3  71.2  79.5  82.15 88.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[41.75 70.15 77.9  84.35 86.45 90.85]%, scope=[1, 3, 5, 9, 11, 19]

(seen / 640d64)
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_640_d64_ln_drop_lamb_seen/111/100k_full 2000 ivfpq-rr 20
matched_exact=[58.25 81.1  86.45 92.4  93.4  96.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[64.75 83.4  87.9  93.1  93.95 96.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[77.5 90.5 93.1 95.4 95.6 97.1]%, scope=[1, 3, 5, 9, 11, 19]

---
<Effect of SNR>
python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_0dbIR/50/10k 2000 ivfpq-rr 20
matched_exact=[76.25 94.9  97.3  98.9  99.15 99.45]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.95 96.1  97.65 99.05 99.25 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.4  98.   98.8  99.45 99.5  99.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_0dbNOIR/36/10k 2000 ivfpq-rr 20
matched_exact=[39.15 77.8  87.9  94.7  96.45 98.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[48.2  81.4  89.3  95.15 96.9  98.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[56.9  85.55 92.3  96.6  97.8  98.6 ]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_5dbIR/50/10k 2000 ivfpq-rr 20
matched_exact=[79.6  95.65 97.55 98.8  99.1  99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[88.1  96.8  97.9  99.   99.35 99.5 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[92.85 98.15 98.9  99.35 99.5  99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_5dbNOIR/36/10k 2000 ivfpq-rr 20
matched_exact=[42.65 79.4  89.15 95.5  96.35 98.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[51.4  82.75 90.9  96.05 96.65 98.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[59.75 86.85 93.45 97.05 97.55 98.85]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_10dbIR/50/10k 2000 ivfpq-rr 20
matched_exact=[78.5  94.5  97.6  98.6  98.95 99.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[85.8  95.75 97.8  98.7  99.3  99.65]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[91.55 97.9  98.8  99.2  99.4  99.65]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_10dbNOIR/36/10k 2000 ivfpq-rr 20
matched_exact=[36.95 76.2  87.35 94.15 95.55 98.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[44.9  79.85 89.05 94.55 95.75 98.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[53.75 83.85 91.25 96.   96.65 98.25]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_20dbIR/50/10k 2000 ivfpq-rr 20
matched_exact=[74.7  92.75 96.4  98.2  98.7  99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[81.85 94.4  96.95 98.35 98.9  99.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[88.   96.5  98.1  98.9  99.2  99.35]%, scope=[1, 3, 5, 9, 11, 19]

python eval_faiss.py cpu logs/emb/exp_NTxent_simCLR_use_anc_rep_120win1_WS_20dbNOIR/36/10k 2000 ivfpq-rr 20
matched_exact=[35.9  75.4  87.   94.1  95.8  97.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[44.2  79.1  88.9  94.6  96.4  98.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[52.25 83.3  91.1  95.8  96.9  98.35]%, scope=[1, 3, 5, 9, 11, 19]


------------------------------------------------------------------------------          
100k-exhaustive: matched=[]                     
100k-ivfpq: matched=[37.08 74.06 85.88 93.46 94.89 97.18]%, scope=[1, 3, 5, 9, 11, 19]

<100k(30s) + skt500k(full)> 
python eval_faiss.py cpu /ssd1/skt_500k_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4_memmap 2000 ivfpq-ondisk 20
DB_SIZE = 231326480
avg length:3.85min
matched_exact=[59.9  89.5  94.   97.2  97.95 98.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[78.95 92.55 95.8  97.85 98.2  99.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.7  94.45 96.8  98.15 98.35 99.25]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full)> 2.7s in-mem
matched_exact=[45.95 76.25 84.8  90.5  91.95 95.2 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[62.4  80.1  86.25 91.3  92.6  95.55]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[74.4  86.65 90.9  94.35 94.9  97.  ]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt500k(full)>
DB_SIZE = 284055920
avg_length = 3.85min
matched_exact=[44.3  75.5  84.5  89.95 91.55 94.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[60.5  79.55 85.9  90.75 92.3  95.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[72.45 86.05 90.55 93.8  94.45 96.9 ]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full)> 0.2s in-mem
matched_exact=[49.3  79.35 87.1  92.3  93.7  95.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[67.   83.7  88.65 93.1  94.4  96.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[81.1  91.15 93.75 96.05 96.65 97.9 ]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt100k(full)> 3s in-mem
matched_exact=[44.95 75.75 84.3  90.   91.65 95.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[60.85 79.7  85.75 90.85 92.35 95.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[72.5  86.   90.45 93.95 94.5  96.85]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt200k(full)> 5s ondisk 12/100 (default)
matched_exact=[44.35 75.5  84.4  90.05 91.65 94.95]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[60.55 79.4  85.85 90.85 92.3  95.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[72.2  85.9  90.55 93.95 94.5  96.85]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt200k(full)> 7s ivfpq-rr, 20/100 
matched_exact=[45.25 76.1  84.65 90.25 91.75 95.05]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[61.35 80.1  86.1  91.   92.45 95.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[72.95 86.4  90.7  93.9  94.6  96.85]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt200k(full)> 7s ivfpq-rr, 40/200 
matched_exact=[45.4  75.5  84.35 89.9  91.25 94.7 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[61.55 79.55 85.8  90.6  91.9  95.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[73.15 86.05 90.4  93.65 94.2  96.75]%, scope=[1, 3, 5, 9, 11, 19]

<100k(full) + skt300k(full)> 5 ondisk 12/100 (default)
matched_exact=[44.85 75.5  84.35 90.15 91.7  95.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[61.15 79.6  85.8  90.9  92.4  95.35]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[73.   86.1  90.55 93.85 94.45 96.85]%, scope=[1, 3, 5, 9, 11, 19]

<50k(full)> 1.5s in-mem
matched_exact=[47.35 77.25 85.6  91.2  92.5  95.6 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[64.5  81.55 87.05 91.95 93.2  95.85]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[77.05 88.3  91.9  94.95 95.5  97.4 ]%, scope=[1, 3, 5, 9, 11, 19]

<10k(full)> in-mem
matched_exact=[49.6  79.4  87.05 92.4  93.75 95.9 ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[67.5  83.8  88.55 93.15 94.4  96.15]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[81.55 91.   93.6  96.1  96.65 97.9 ]%, scope=[1, 3, 5, 9, 11, 19]

<5k(full)> 0.2s in-mem
matched_exact=[50.2 79.6 87.3 92.5 93.7 96. ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[68.15 84.05 88.95 93.2  94.3  96.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[82.55 91.5  94.2  96.2  96.75 97.95]%, scope=[1, 3, 5, 9, 11, 19]

<2k(full)>
matched_exact=[50.8  80.05 87.75 92.4  93.75 96.  ]%, scope=[1, 3, 5, 9, 11, 19]
matched_near=[69.   84.45 89.35 93.1  94.4  96.25]%, scope=[1, 3, 5, 9, 11, 19]
matched_song=[84.05 92.1  94.6  96.3  96.85 97.95]%, scope=[1, 3, 5, 9, 11, 19]


---
<100k(full) re>
python eval_faiss.py cpu logs/emb/exp_v2fix_semihard_320win1_d64_tr100k_100kfull/4 2000 ivfpq-rr 20

@author: skchang@cochlear.ai
"""
import sys, os
import numpy as np
import faiss
import time
from scipy.special import softmax
import tensorflow as tf
#from utils.Faiss_ivf_ondisk_util import index_ivf_read_write_ondisk

SEL_POS_QUERY = np.arange(0, 140000) #1198000) #49980)
# SEL_NEG_QUERY = np.arange(1198020, 2395900)#50000, 99940)
"""
emb_dir = 'logs/emb/exp_v2fix_semihard_320win1_d64_tr100k_100k/4'
scope = [1, 3, 5, 9, 11, 19]
mode = 'ivfpq-ondisk'
n_probe=20
EMB_DIR = './temp_emb_dir'
emb_dir = EMB_DIR

EMB_DIR = '/ssd1/skt_50M_emb/exp_v2fix_semihard_320win1_d64_tr100k_191206-1347/4_memmap'
SEARCH_MODE=mode
"""
# Arguments parser
if __name__ == "__main__":
    if len(sys.argv) != 1:
        if len(sys.argv) > 1:
            if str.upper(sys.argv[1]) == 'GPU':
                USE_GPU = True
            else:
                USE_GPU = False
        
        if len(sys.argv) > 2:
            EMB_DIR = sys.argv[2]
        else:
            raise KeyError('<argv_1> for embedding directory is required!!')
        
        if len(sys.argv) > 3:
            N_SAMPLE_TEST = int(sys.argv[3])
        else:
            N_SAMPLE_TEST = 2000
        
        if len(sys.argv) > 4:
            assert (sys.argv[4] in ['l2', 'ivf', 'pq', 'ivfpq', 'pq-rr', 'ivfpq-rr','ivfpq-rr128', 'ivfpq-rr32','ivfpq-rr16','ivfpq-rr8', 'ivfpq-ondisk', 'hnsw'])
            SEARCH_MODE = sys.argv[4]
        else:
            SEARCH_MODE = 'l2'
        
        if len(sys.argv) > 5:
            N_PROBE = int(sys.argv[5])
        else:
            N_PROBE = 20
    
        TEST_MIREX_QUERY = False
        if len(sys.argv) > 6:
            if 'MIREX' in str.upper(sys.argv[6]):
                TEST_MIREX_QUERY = True
    
        USE_SOFTMAX = False
        if len(sys.argv) > 7:
            if 'SOFTMAX' in str.upper(sys.argv[7]):
                USE_SOFTMAX = True
    else:
        USE_SOFTMAX = False
    


# MAIN
def eval(use_gpu=False,
         emb_dir=str(),
         n_sample_test=4000,
         scope=[1, 3, 5, 9, 11, 19],
         mode='l2',
         on_disk=False,
         n_probe=20,
         display_interval=1,
         test_mirex_query=False,
         use_softmax=False,
         transformer=None):
    # Print summary
    print('\n-------summary--------')
    print('search mode:{}'.format(mode))
    print('n_probe:{}'.format(n_probe))
    print('test_mirex:{}'.format(str(test_mirex_query)))
    print('use_softmax:{}'.format(str(use_softmax)))
    if transformer is not None:
        print('transformer:True')
    else:
        print('transformer:False')
    
    
    
    # Get filepaths
    db_fpath = emb_dir + '/db.mm'
    query_fpath = emb_dir + '/query.mm'
    
    if test_mirex_query:
        mirex_db_fpath = emb_dir + '/../MIREX_db.npy'
        mirex_query_fpath = emb_dir + '/../MIREX_query.npy'
    

    # Get shapes
    db_shape = np.load(emb_dir + '/db_shape.npy')
    query_shape = np.load(emb_dir + '/query_shape.npy') # not used with MIREX
    ''' 
    TEST 20191226
    '''
    #db_shape[0] = int((284055920//6)//10//3.95)

    # Load DB and queries from memory-mapped-disk
    db = np.memmap(
        db_fpath, dtype='float32', mode='r',
        shape=(db_shape[0], db_shape[1]))  # (nItem, dim)
        
    
    if test_mirex_query:
        # Test with MIREX queries within {our DB + MIREX DB}
        query = np.load(mirex_query_fpath)
        query_shape = query.shape
        query_pos = query
        db_ext = np.load(mirex_db_fpath)
        db = np.concatenate((db_ext, db), axis=0)
        # n_sample_test = query.shape[0] - 1
        n_sample_test = query.shape[0] // 19 # test for every 10s
    else:
        # Test with our queries
        query = np.memmap(
            query_fpath,
            dtype='float32',
            mode='r',
            shape=(query_shape[0], query_shape[1],
                   query_shape[2]))  # (nItem, nAug, dim)
    
        query_pos = query[SEL_POS_QUERY, 0, :]  #[N,D]
        #query_neg = query[SEL_NEG_QUERY, 0, :]
        # delete negative query from DB
        #SEL_DB = np.r_[SEL_POS_QUERY, (SEL_NEG_QUERY[-1] + 20):db.shape[0]]
        #db = db[SEL_DB, :]
    print(f'DB shape: {db_shape}\nQUERY shape: {query_shape}')
    
    # Indexing FAISS DB
    d = int(db_shape[1])  # dimension of embeddings
    if on_disk:
        pass
    else:        
        index = faiss.IndexFlatL2(d)    
        
        if mode == 'l2':
            index.add(db)
        elif mode == 'ivf':
            nlist = 400  # It clusters all input vectors into nlist groups (nlist is a field of IndexIVF). At add time, a vector is assigned to a groups. At search time, the most similar groups to the query vector are identified and scanned exhaustively
            index = faiss.IndexIVFFlat(index, d, nlist)
            index.train(db)
            index.add(db)
        elif mode == 'pq':
            M = 16
            nlist = 8
            index = faiss.IndexPQ(d, M, nlist)
            index.train(db)
            index.add(db)
        elif mode == 'ivfpq':
            code_sz = 64  # power of 2: 4~64
            n_centroids = 100  #100 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
            nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
            n_probe_centroids = 4 # This is important. increasing this will help accuracy, and lower speed.
            index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
            index.train(db)
            index.add(db)
            index.nprobe = n_probe_centroids
        elif mode == 'ivfpq-rr':
            #index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits, M_refine, nbits_refine)
            if use_gpu:
                print('use_GPU: True')
                code_sz = 64 #64  # power of 2: 4~64
                n_centroids = 256#262144#200  #100 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
                nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
                n_probe_centroids = 40 #40#12
                M_refine = 4
                nbits_refine = 4
                index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits, M_refine, nbits_refine)
                #index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
                #index2 = faiss.index_factory(128, "IVF262144, PQ4+4")
                GPU_RESOURCES = faiss.StandardGpuResources()
                GPU_OPTIONS = faiss.GpuClonerOptions()
                GPU_OPTIONS.useFloat16 = True # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
                #GPU_OPTIONS.usePrecomputed = False
                #GPU_OPTIONS.indicesOptions = faiss.INDICES_CPU
                index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index,
                                               GPU_OPTIONS)
            else:
                print('use_GPU: False')
                code_sz = 64  # power of 2: 4~64
                n_centroids = 200  #100 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
                nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
                n_probe_centroids = 40 #40#12
                M_refine = 4
                nbits_refine = 4
                index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits, M_refine, nbits_refine)  
            
            if len(db) > 10223616:
                index.train(db[:10223616,:])
            else:
                index.train(db)
            index.add(db)
            index.nprobe = n_probe_centroids
        elif mode == 'ivfpq-milvus':
            pass;
        elif mode == 'ivfpq-ondisk':
            code_sz = 64  # power of 2: 4~64
            n_centroids = 100  #100 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
            nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
            n_probe_centroids = 20 #12
            #M_refine = 4
            #nbits_refine = 4
            #index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits, M_refine, nbits_refine)
            index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits) #, M_refine, nbits_refine)
            if USE_GPU:
                index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index)
            else:
                from utils.Faiss_ivf_ondisk_util import index_ivf_read_write_ondisk
                index = index_ivf_read_write_ondisk(index=index,
                                                    mode='w', #'w',
                                                    emb_dir=EMB_DIR,
                                                    emb_all=db,
                                                    n_part=2)
            index.nprobe = n_probe_centroids
        elif mode == 'pq-rr':
            M = 16
            nlist = 8
            index_pq = faiss.IndexPQ(d, M, nlist)
            index = faiss.IndexRefineFlat(index_pq)
            index.train(db)
            index.add(db)
            index.k_factor = 4 # or 4
        elif mode == 'hnsw':
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            # training is not needed; higher is more accurate and slower to construct.
            index.hnsw.efConstruction = 80 #40
            index.verbose = True
            index.add(db)
            index.hnsw.search_bounded_queue = True
        else:
            raise NotImplementedError(mode)

        assert (index.is_trained)
        print('\nIndex is succesfully trained in mode: ' + mode)
        print('nTotalDB: ', index.ntotal)

    # Remove neg queries for recall-test
    #index.remove_ids(SEL_NEG_QUERY)
    #print('Successfully removed {} items from DB'.format(len(SEL_NEG_QUERY)))
    #print('nTotalDB: ', index.ntotal)
    
    # Search
    if test_mirex_query:
        n_test = n_sample_test
    else:
        n_test = len(query_pos) - np.max(scope) + 1
    matched_pos_exact = np.zeros((n_sample_test,
                                  len(scope))).astype(np.int)  # True Positive
    matched_pos_near = np.zeros((n_sample_test, len(scope))).astype(np.int)  #
    matched_pos_song = np.zeros((n_sample_test, len(scope))).astype(np.int)
    confidence_pos = np.zeros((n_sample_test, len(scope)))  # distances

    # test with positive queries:
    start_time = time.time()
    
    if test_mirex_query:
        # t_indices = np.arange(n_test)
        t_indices = np.arange(0, n_test * 19, 19)
    else:
        np.random.seed(0)
        t_indices= np.random.permutation(n_test)[:n_sample_test]
    
    for ti, t in enumerate(t_indices):  # b: batch index
        for si, s in enumerate(scope):  # si: scope_index, s:scope_value
            q = query_pos[t:(t + s), :]  # set of query with length=s (s, dim)

            D, I = index.search(
                q, n_probe
            )  # D: distance, I: result index, n_probe: top_k will be returned

            # Adjust offsets
            for offset in range(len(I)):
                I[offset, :] -= offset
            # Collect canditates ids:
            candidates = np.unique(I[np.where(I >= 0)])  # ignore id < 0

            
            if transformer == None:
                # Calculate average Sqaured L2 Distance of each candidates
                _dists = np.zeros(len(candidates))
                for ci, cid in enumerate(candidates):
                    if use_softmax==True:
                        dot_prod = 2 * (1 - np.dot(q[:, :], db[cid:(cid + s), :].T))
                        importance = np.sum(softmax(dot_prod, 1), 0)
                        seq_score = np.diag(dot_prod) * importance
                        _dists[ci] = np.mean(seq_score)
                    else:
                        _dists[ci] = np.mean(
                            np.diag(2 * (1 - np.dot(q[:, :], db[cid:(cid + s), :].T))))
                pred_id = candidates[np.argmin(_dists)]
                
                # if pred_id == t:  # t is gt id
                #     matched_pos_exact[ti, si] = 1
                # # else:
                # #     print(t, _dists[np.argmin(_dists)], np.sort(_dists), candidates[np.argsort(_dists)])
                
                # if test_mirex_query: # MIREX test allows 2s tolerance
                #     if pred_id in np.arange(t - 4, t + 5):
                #         matched_pos_near[ti, si] = 1
                # else:
                #     if pred_id in [t - 1, t, t + 1]:
                #         matched_pos_near[ti, si] = 1
                
                # if test_mirex_query: # MIREX test allows 2s tolerance
                #     if np.abs(pred_id - t) < 62:
                #         matched_pos_song[ti, si] = 1
                # else:
                #     if np.abs(pred_id - t) < 600:
                #         matched_pos_song[ti, si] = 1
    
                # confidence_pos[ti, si] = min(_dists)
            else:
                # 2020. Nov. Search with Transformer
                _dists = np.zeros(len(candidates))
                targets = []
                inputs = []
                for ci, cid in enumerate(candidates):
                    if (cid+s+s-1) > len(db):
                        cid = cid - 100 # NOTE: this is temporary solution to avoid out of bounds error!
                    _target = db[(cid + s - 1), :] # target to compare: (dim,)
                    _input_q = q[:,:] 
                    _input_q[-1,:] = 0. # mask current with zeros 
                    _input_db = db[(cid+s):(cid+s+s-1), :]
                    _input = np.concatenate((_input_q, _input_db), axis=0)
                    
                    targets.append(_target)
                    inputs.append(_input)
                targets = np.stack(targets) # (nCandidates, dim)
                inputs = tf.constant(np.stack(inputs)) # # (nCandidates, total_scope, dim)
                
                outputs = transformer.context_fp(inputs, training=False, mask=transformer.att_mask)
                # take center outputs only
                outputs = outputs[:, (s-1), :].numpy() # (nCandidates, dim)
                
                # calculate distances
                for ci, cid in enumerate(candidates):
                    _dists[ci] = np.mean(
                        np.diag(2 * (1 - np.dot(targets, outputs.T))))
                pred_id = candidates[np.argmin(_dists)]
            
            
            
            if pred_id == t:  # t is gt id
                matched_pos_exact[ti, si] = 1
            # else:
            #     print(t, _dists[np.argmin(_dists)], np.sort(_dists), candidates[np.argsort(_dists)])
            
            if test_mirex_query: # MIREX test allows 2s tolerance
                if pred_id in np.arange(t - 4, t + 5):
                    matched_pos_near[ti, si] = 1
            else:
                if pred_id in [t - 1, t, t + 1]:
                    matched_pos_near[ti, si] = 1
            
            if test_mirex_query: # MIREX test allows 2s tolerance
                if np.abs(pred_id - t) < 62:
                    matched_pos_song[ti, si] = 1
            else:
                if np.abs(pred_id - t) < 600:
                    matched_pos_song[ti, si] = 1

            confidence_pos[ti, si] = min(_dists)
                

        if (ti != 0) & ((ti % display_interval) == 0):
            elapsed_time = time.time() - start_time
            start_time = time.time() # reset start time
            print('elapse time per {} iterations: {:.4f} sec.'.format(display_interval, elapsed_time))
            
            _acc_exact = 100 * np.mean(matched_pos_exact[:ti + 1, :], axis=0)
            _acc_near = 100 * np.mean(matched_pos_near[:ti + 1, :], axis=0)
            _acc_song = 100 * np.mean(matched_pos_song[:ti + 1, :], axis=0)
            print('{}/{}\nmatched_exact={}%, scope={}'.format(ti, n_sample_test, np.around(
                    _acc_exact, decimals=2), scope))
            print('matched_near={}%, scope={}'.format(np.around(
                    _acc_near, decimals=2), scope))
            print('matched_song={}%, scope={}'.format(np.around(
                    _acc_song, decimals=2), scope))

    acc_exact = 100 * np.mean(matched_pos_exact[:ti + 1, :], axis=0)
    acc_near = 100 * np.mean(matched_pos_near[:ti + 1, :], axis=0)
    acc_song = 100 * np.mean(matched_pos_near[:ti + 1, :], axis=0)

    # Write(append) results to file
    result_path = emb_dir + '/../result.txt'
    task = os.path.dirname(emb_dir+'/').split('/')[-1]
    task += 'MIREX' if test_mirex_query else ''
    if transformer is not None:
        task += '+ transformer' 
    f = open(result_path, 'a')
    f.write(f'task={task}, use_gpu={use_gpu}, mode={mode}, n_probe{n_probe}, n_test{n_sample_test}\n')
    f.write('elapse time per {} iterations: {:.4f} sec.\n'.format(display_interval, elapsed_time))
    f.write('{}/{}\nmatched_exact={}%, scope={}\n'.format(ti, n_sample_test, np.around(acc_exact, decimals=2), scope))
    f.write('matched_near={}%, scope={}\n'.format(np.around(acc_near, decimals=2), scope))
    f.write('matched_song={}%, scope={}\n'.format(np.around(acc_song, decimals=2), scope))
    f.write('\n')
    f.close()
    print(f'---Saved evaluation result to {result_path} succesfully...---')
    #save_results()
    return




if __name__ == "__main__":
    # execute only if run as a script
    eval(use_gpu=USE_GPU, emb_dir=EMB_DIR, n_sample_test=N_SAMPLE_TEST,
         mode=SEARCH_MODE, n_probe = N_PROBE, display_interval=10,
         test_mirex_query=TEST_MIREX_QUERY, use_softmax=USE_SOFTMAX)

#%timeit -n 3 test1(q, n_probe)
#%timeit -n 3 test2(scope=[3], n_probe=100)
# test2 scope[1,3,5,9,19], n_probe=5 --> [94.5, 102, 119, 213, 343] ms
# test2_scope[1,3,5,9,19], n_probe=10 --> [215]
# test2_scope[1,3,5,9,19], n_probe=100 --> [102, 105, 231, 365]
