import strlearn as sl
import numpy as np


# Generate data streams
n_chunks = 3000
chunk_size = 250
# Imabalance
weights = [[.85, .15], [.95, .05]]
y_flips =[.01, .05, .15]
"""
1. n_drifts
2. concept_sigmoid_spacing (None for sudden)
3. incremental [True] or gradual [False]
4. recurring [True] or non-recurring [False]
"""
# Sudden, Gradual, Incremental
drifts = [(30, None, False, False), (30, 5, False, False), (30, 5, True, False)]
"""
Concept:
1. n_features
2. n_informative
3. n_redundant
4. n_repeated
5. n_clusters_per_class
"""
concepts = [(8, 8, 0, 0, 1)]

def generate_imb_streams(random_state, replications):
    random = np.random.RandomState(random_state)
    random_states = random.randint(0, 99999, replications)
    
    streams = []
    for concept in concepts:
        for weight in weights:
            for y_flip in y_flips:
                for drift in drifts:
                    for random in random_states:
                        stream = sl.streams.StreamGenerator(n_chunks=n_chunks, chunk_size=chunk_size, weights=weight,y_flip=y_flip,
                                                            n_drifts=drift[0], concept_sigmoid_spacing=drift[1], incremental=drift[2], recurring=drift[3], 
                                                            n_features=concept[0], n_informative=concept[1], n_redundant=concept[2], n_repeated=concept[3], n_clusters_per_class=concept[4],
                                                            random_state=random)
                        streams.append(stream)
    return streams

# 265, 359
def realstreams():
    return {
        "covtypeNorm-1-2vsAll": sl.streams.ARFFParser("real_streams/covtypeNorm-1-2vsAll-pruned.arff", n_chunks=265, chunk_size=1000),
        "poker-lsn-1-2vsAll": sl.streams.ARFFParser("real_streams/poker-lsn-1-2vsAll-pruned.arff", n_chunks=359, chunk_size=1000),
    }