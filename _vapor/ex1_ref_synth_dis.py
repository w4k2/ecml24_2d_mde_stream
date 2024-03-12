import strlearn as sl
from skmultiflow.trees import HoeffdingTree
from skmultiflow.meta import LearnNSE
import numpy as np
from utils import generate_streams, CDS, Discretizer
from strlearn.metrics import balanced_accuracy_score as bac, recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2
from strlearn.ensembles import SEA, AWE, AUE, WAE, KUE, ROSE, NIE
import multiprocessing
from sklearn.naive_bayes import GaussianNB
from utils import TestThenTrainDis

random_state = 1410
replications = 5
n_estimators = 10

streams = generate_streams(random_state, replications)

results = []

def worker(stream):
    print("Start: %s" % (stream))
    eval = sl.evaluators.TestThenTrain(metrics=(recall, precision, specificity, f1_score, geometric_mean_score_1, geometric_mean_score_2, bac), verbose=False)
    methods = [
        HoeffdingTree(split_criterion="hellinger"),
        # SEA(clf, n_estimators),
        # AWE(clf, n_estimators),
        # AUE(clf, n_estimators),
        CDS(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        NIE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        KUE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        ROSE(HoeffdingTree(split_criterion="hellinger"), n_estimators),
        # LearnNSE(clf, n_estimators=n_estimators)
    ]
    eval.process(stream, methods)
    np.save("results/ref_synth/%s_discrete" % stream, eval.scores)
    print("End: %s" % (stream))
    
jobs = []
if __name__ == '__main__':
    for stream in streams:
        p = multiprocessing.Process(target=worker, args=(stream,))
        jobs.append(p)
        p.start()