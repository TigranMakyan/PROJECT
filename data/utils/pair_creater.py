import numpy as np
import os
import random
random.seed(10)

map5 = 'map_0.05'
map10 = 'map_0.10'
map20 = 'map_0.20'
map40 = 'map_0.40'
COUNT = 500
rel_scale = 1.0
for i, image_i in enumerate(random.sample(os.listdir('/home/user/Datasets/map/map_0.40/'), 250)):
    #rev_folder = os.listdir('/home/user/Datasets/map/map_0.20/')[::-1]
    for j, image_j in enumerate(random.sample(os.listdir('/home/user/Datasets/map/map_0.40/'), 250)):
        #if j == 500: break
        print(f'{i} image_i and {j} image_j')
        with open('/home/user/Datasets/map/pairs_sampled.txt', 'a') as f:
            f.write(f'{os.path.join(map40, image_i)} {os.path.join(map40, image_j)} {rel_scale}')
            f.write('\n')



# 0.25 0.5 0.125