import json
import numpy as np
# test = np.load('tang.npz')
# data, ix2word, word2ix = test['data'], test['ix2word'].item(), test['word2ix'].item()


# with open('dic.json','a') as outfile:
#     dic=  {'word2ix':word2ix,"ix2word":ix2word}
#     json.dump(dic,outfile)
#     outfile.write('\n')


res = 1.0
for i in range(51,101):
    res-= 1/i
print(res)

import tensorflow