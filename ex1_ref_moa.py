import strlearn as sl
from skmultiflow.trees import HoeffdingTree
import numpy as np
from utils import moa_streams, CDS
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from strlearn.ensembles import KUE, ROSE, NIE
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score
import multiprocessing

n_estimators = 10

streams = moa_streams()

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
    np.save("results/ref_moa/%s" % stream, eval.scores)
    print("End: %s" % (stream))
    
jobs = []
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=18)
    for stream in streams.keys():
        pool.apply_async(worker, args=(stream,))
    pool.close()
    pool.join()