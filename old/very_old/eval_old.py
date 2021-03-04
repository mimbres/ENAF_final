EXP_NAME = 'exp_all-balanced'
EPOCH = ''
TEST_MODE = 'test-only'

Xa, Xp = test_ds.__getitem__(102)



#%%
import matplotlib.pyplot as plt
plt.figure(); plt.imshow(emb)


emb_a, emb_p = tf.split(emb, [TS_N_ANCHOR, TS_BATCH_SZ - TS_N_ANCHOR], axis=0)
plt.figure(); plt.imshow(emb_a)
plt.figure(); plt.imshow(emb_p)



Online_Triplet_Loss_ts = Online_Batch_Triplet_Loss(
    TS_BATCH_SZ,
    TS_N_ANCHOR,
    n_pos_per_anchor=int((TS_BATCH_SZ - TS_N_ANCHOR) / TS_N_ANCHOR),
    use_anc_as_pos=True)

total_samples = db.shape[0]
#%%
i = 104
que = query[i, :, :]
dists = Online_Triplet_Loss_ts._pairwise_distances_v2(que, db, 
                                                      use_anc_as_pos=False,
                                                      squared=True)
_, ranks = np.where(np.argsort(dists, axis=1)==100)
print(ranks)


#%% rank by sum(dists) of 8 samples in a row 
total_samples = db.shape[0]
sum_dists = np.zeros((4, total_samples))
for i in range(1000, 1000+8):
    que = query[i, :, :]
    dists = Online_Triplet_Loss_ts._pairwise_distances_v2(que, db, 
                                                      use_anc_as_pos=False,
                                                      squared=True)
    sum_dists = sum_dists + dists
    
ranks = np.argsort(sum_dists, axis=1)
print(ranks)
np.where(ranks==1000)


#%% sum distances in succession using diagonal kernel
total_samples = db.shape[0]
SCOPE = 9 # better using odd numbers
n_test = total_samples - SCOPE

_sum_ranks = 0.
n_corrects = np.zeros(3) # top1, top3, top10
progbar = tf.keras.utils.Progbar(n_test)
#for target_id in range(n_test):
for cnt, target_id in enumerate(np.random.permutation(n_test)):    
    # Collects distances in succession within SCOPE
    all_dists = np.zeros((4, total_samples, SCOPE, 1))
    for i in range(SCOPE):
        que = query[i + target_id, :, :]
        dists = Online_Triplet_Loss_ts._pairwise_distances_v2(que, db, 
                                                          use_anc_as_pos=False,
                                                          squared=True).numpy()
        all_dists[:, :, i, 0] = dists
        
        
    # Convolve with eye kernel
    conv_eye = tf.keras.layers.Conv2D(filters=1, kernel_size=[SCOPE, SCOPE],
        padding='valid', use_bias=False,
        kernel_initializer=tf.constant_initializer(np.eye(SCOPE).reshape((SCOPE, SCOPE, 1, 1))))
    conv_dists = conv_eye(all_dists).numpy() # [4, 14296, 1, 1]
    conv_dists = np.squeeze(conv_dists) # (4, 14296)
    
    # Mean-Rank
    sorted = np.argsort(conv_dists, axis=1)
    #print(ranks)
    _, ranks = np.where(sorted==target_id)
    _sum_ranks += sum(ranks) 
    
    # Top N
    n_corrects += [sum(ranks<1), sum(ranks<4), sum(ranks<11)]    
    progbar.add(1, values=[("top1_acc",  (sum(ranks<1)/4)*100. )])
    tf.print()

#mean_rank = _sum_ranks / (n_test * 4) + 1
#top_n_acc = (n_corrects / (n_test * 4)) * 100.
mean_rank = _sum_ranks / ((cnt+1) * 4) + 1
top_n_acc = (n_corrects / ((cnt+1) * 4)) * 100.

tf.print('\n mean_rank = {}\n top_1_acc = {}\n top_3_acc = {}\n top_10_acc = {}'.format(
        mean_rank, top_n_acc[0], top_n_acc[1], top_n_acc[2]))

"""
500 songs (30s), (60,000queries) scope=9(10s, 9 segments with 2s win and 1s overlap)
 mean_rank = 2.0667191325638337
 top_1_acc = 96.4603008044771
 top_3_acc = 98.9401888772298
 top_10_acc = 99.38789786638685
 
scope=7
 mean_rank = 2.8448450723928094
 top_1_acc = 94.31174372245927
 top_3_acc = 98.18668252080856
 top_10_acc = 98.92110232915996
 
scope=5
 mean_rank = 4.689960836422127
 top_1_acc = 89.86467585145814
 top_3_acc = 96.2427442478495
 top_10_acc = 97.69214630393734

scope=3
 mean_rank = 12.138486819103559
 top_1_acc = 76.53310957275716
 top_3_acc = 89.80665687714145
 top_10_acc = 93.76442206838682

scope=1
 mean_rank = 74.10440117457875
 top_1_acc = 31.563308396839822
 top_3_acc = 56.73634901768859
 top_10_acc = 70.62853946724464


10,000 songs (30s) (200,000 queries) scope=9(10s, 9 segments with 2s win and 1s overlap), 
 mean_rank = 65.58951298500382
 top_1_acc = 93.00006188246456
 top_3_acc = 95.70277852265929
 top_10_acc = 96.64906454341055

scope=7
 mean_rank = 85.96392240359593
 top_1_acc = 89.56115448308492
 top_3_acc = 93.86779433798597
 top_10_acc = 95.36412743474489
 
scope=5
 mean_rank = 141.4327920210973
 top_1_acc = 82.44998314595603
 top_3_acc = 89.88806931969148
 top_10_acc = 92.41221025915571
 
scope=3
 mean_rank = 271.0820203210131
 top_1_acc = 64.76770725960831
 top_3_acc = 78.68778530407893
 top_10_acc = 84.30183330879105
 
scope=1
 mean_rank = 1075.5179391157285
 top_1_acc = 16.76582507852138
 top_3_acc = 35.191169364580816
 top_10_acc = 48.394841749214784

"""

