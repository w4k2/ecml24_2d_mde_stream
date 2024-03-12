import strlearn as sl
from skmultiflow.trees import HoeffdingTree
from skmultiflow.meta import LearnNSE
import numpy as np
from utils import realstreams, CDS
from strlearn.metrics import balanced_accuracy_score as bac
from strlearn.ensembles import SEA, AWE, AUE, WAE, KUE, ROSE, NIE
import multiprocessing
from sklearn.naive_bayes import GaussianNB

random_state = 1410
n_estimators = 10

streams = realstreams()

results = []

def worker(stream):
    print("Start: %s" % (stream))
    clf = HoeffdingTree(split_criterion="hellinger")
    # clf = GaussianNB()
    eval = sl.evaluators.TestThenTrain(metrics=(bac), verbose=True)
    methods = [
        clf,
        # SEA(clf, n_estimators),
        # AWE(clf, n_estimators),
        # AUE(clf, n_estimators),
        CDS(clf, n_estimators),
        NIE(clf, n_estimators),
        WAE(clf, n_estimators),
        KUE(clf, n_estimators),
        ROSE(clf, n_estimators),
        # LearnNSE(clf, n_estimators=n_estimators)
    ]
    eval.process(streams[stream], methods)
    np.save("results/ref_real/%s" % stream, eval.scores)
    print("End: %s" % (stream))
    
jobs = []
if __name__ == '__main__':
    for stream in streams:
        p = multiprocessing.Process(target=worker, args=(stream,))
        jobs.append(p)
        p.start()