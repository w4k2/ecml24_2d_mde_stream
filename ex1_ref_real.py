import strlearn as sl
from skmultiflow.trees import HoeffdingTree
import numpy as np
from utils import realstreams, CDS
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from strlearn.ensembles import KUE, ROSE, NIE
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score
import multiprocessing

random_state = 1410
n_estimators = 10

streams = realstreams()
s = streams["INSECTS-abrupt_imbalanced_norm"]
for c in range(300):
    print(c)
    X, y = s.get_chunk()
    print(X)
exit()

def worker(stream):
    print("Start: %s" % (stream))
    eval = sl.evaluators.TestThenTrain(metrics=(recall, recall_score, precision, precision_score, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac, balanced_accuracy_score), verbose=False)
    methods = [
        HoeffdingTree(split_criterion="hellinger"),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
    ]
    eval.process(streams[stream], methods)
    np.save("results/ref_real/%s" % stream, eval.scores)
    print("End: %s" % (stream))
    
jobs = []
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10)
    for stream in streams.keys():
        pool.apply_async(worker, args=(stream,))
    pool.close()
    pool.join()