from jiwer import wer
import numpy as np
import Levenshtein



h = np.load('../predictions_25.npy')
r = np.load('../true_25.npy')

h = h[:7000]
r = r[:7000]

r = list(r)
h = list(h)

S = 0
D = 0
I = 0
H = 0
N = 0


#r = ['This is a reference']
#h = ['This is a reference and']

for i in range(len(r)):
    editops = Levenshtein.editops(r[i], h[i])
    S += sum(1 if op[0] == "replace" else 0 for op in editops)
    D += sum(1 if op[0] == "delete" else 0 for op in editops)
    I += sum(1 if op[0] == "insert" else 0 for op in editops)
    N += len(r[i].split())


wer_res = ((S + D + I) / N)

print(wer_res * 100)
print(wer(r, h) * 100)




