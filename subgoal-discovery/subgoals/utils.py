import numpy as np
import glob

def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + '-results.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' %item
    string_to_write += ' ' + '%.2f' %cmap
    fid.write(string_to_write + '\n')
    fid.close()

def get_labels(seq_len, n_subgoals):
    # Equi-partition labels
    stops = np.array(range(1,n_subgoals+1)).astype('float32')/n_subgoals
    labels = np.zeros((seq_len, len(stops)), dtype=float)
    prev_idx = 0
    for i, stop in enumerate(stops):
        idx = int(seq_len*stop)
        labels[prev_idx:idx, i] = 1.
        prev_idx = idx
    return labels

def dist(a,b):
    return np.sum(np.abs(a-b))

def DTW(seq1, seq2):
    D = np.zeros((len(seq1), len(seq2)), dtype='float')  
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            D[i,j] = dist(seq1[i,:],seq2[j,:])
            if i-1 < 0 and j-1 < 0:
                continue
            elif i-1 < 0:
                D[i,j] += D[i,j-1]
            elif j-1 < 0:
                D[i,j] += D[i-1,j]
            else:
                D[i,j] += min(D[i-1,j-1], D[i-1,j], D[i, j-1])
    
    A = np.zeros((len(seq1), len(seq2)))
    i, j = len(seq1)-1, len(seq2)-1
    A[i,j] = 1
    while True:
        if i-1 < 0 and j-1 < 0:
            break
        elif i-1 < 0:
            A[i,j-1] = 1; j -= 1
        elif j-1 < 0:
            A[i-1,j] = 1; i -= 1
        else:
            if D[i-1,j-1] <= D[i-1,j]  and D[i-1,j-1] <= D[i,j-1]:
                A[i-1,j-1] = 1; i -= 1; j -= 1
            elif D[i-1,j] <= D[i-1,j-1]  and D[i-1,j] <= D[i,j-1]:
                A[i-1,j] = 1; i -= 1
            else:
                A[i,j-1] = 1; j -= 1

    return A

def getAssignment(seq, n_class):
    ''' seq1 is of dimension n_steps x n_class '''
    return DTW(seq, np.eye(n_class, dtype='float'))

        
