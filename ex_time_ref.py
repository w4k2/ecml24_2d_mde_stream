import strlearn as sl
from skmultiflow.trees import HoeffdingTree
from skmultiflow.meta import LearnNSE
import numpy as np
from utils import generate_imb_streams, CDS
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from strlearn.ensembles import SEA, AWE, AUE, WAE, KUE, ROSE, NIE
from time import time

n_estimators = 10
n_chunks=110
chunk_size=250
n_features = 64

stream = sl.streams.StreamGenerator(n_chunks=n_chunks, chunk_size=chunk_size, n_features=n_features, n_informative=n_features, n_redundant=0, random_state=1410, n_drifts=1, weights=[.9, .1])

methods = [
    HoeffdingTree(split_criterion="hellinger"),
    CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
]

time_results = np.zeros((5, 110))

print("Start: %s" % (stream))
for chunk_id in range(n_chunks):
    
    X, y = stream.get_chunk()
    
    for method_id, method in enumerate(methods):
        start = time()
        if chunk_id == 0:
            method.partial_fit(X, y)
        else:
            pred = method.predict(X)
            method.partial_fit(X, y)
            
        elapsed = time() - start
        time_results[method_id, chunk_id] = elapsed
        
            
np.save("results/time/time_ref_64", time_results)
print("End: %s" % (stream))