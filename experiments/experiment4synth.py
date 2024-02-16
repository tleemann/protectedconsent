import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.synthetic_data import sample_continuous_dataset, sample_binary_dataset, binary_ground_truth_off
from utils.synthetic_data import sample_missing_features, induce_multiple_independent_missingness

from models.fairmiss_regression import ContinousLogisticRegression, NonFairContinousLogisticRegression
from models.multi_model import FairMultiModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from utils.data_utils import aggregate_scores, evaluate_and_append_scores
from utils.pucida import exhaustive_resampling, random_resampling
from models.non_parametric import NonParametricRegression
import json

n_samples_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
# Set parameters of the data generating processes
## Process 2: Binary Variables
n_features = 8

p_matrix = np.array([[0.09059388, 0.14105619],
       [0.91516404, 0.93091497],
       [0.22575713, 0.02090318],
       [0.77195219, 0.3771833 ],
       [0.20264837, 0.34734818],
       [0.96890756, 0.32246523],    
       [0.8748638 , 0.23904882],
       [0.7239531 , 0.15963419],])
# This matrix was once randomly drawn but is now fixed (with only order of features changed)

p_y1 = 0.6 # prior for y=1

p_missing_y0 = np.array([0.92075329, 0.64741526, 0.50827616])
p_missing_y1 = np.array([0.34541013, 0.29449127, 0.20754432])
p_missing = np.vstack((p_missing_y0, p_missing_y1)).T
f_missingness = np.array([5,6,7])


def sample_binary_data(n_samples):
    X, Y = sample_binary_dataset(p_matrix, p_y1, n_records=n_samples)
    Xm, M = induce_multiple_independent_missingness(X, Y, f_missingness, p_missing_y0 ,p_missing_y1)
    return Xm[:, :-len(p_missing_y0)], Xm[:, -len(p_missing_y0):], 1-M, Y

def binary_off_proba_gt(X, A):
    Xmod = X.copy()
    Xmod[:,-A.shape[1]:] = X[:,-A.shape[1]:]-(1-A)
    py1 = binary_ground_truth_off(Xmod, p_y1, p_matrix, p_missing, off=True)
    return np.stack((1.0-py1, py1)).T

def binary_unfair_proba_gt(X, A):
    Xmod = X.copy()
    Xmod[:,-A.shape[1]:] = X[:,-A.shape[1]:]-(1-A)
    py1 = binary_ground_truth_off(Xmod, p_y1, p_matrix, p_missing, off=False)
    return np.stack((1.0-py1, py1)).T

process_2 = ("binary", sample_binary_data, binary_off_proba_gt, binary_unfair_proba_gt)


# Classifiers with their args.

n_test = 5000
n_reps = 5

classifier_class_l = [
                        (NonParametricRegression, {}),
                        (RandomForestClassifier, {"n_estimators": 100}),
                        (ExtraTreesClassifier, {"n_estimators": 100, "min_samples_split": 10}),
                        (DecisionTreeClassifier, {"min_samples_leaf": 30}),
                        (GradientBoostingClassifier, {"min_samples_leaf": 30, "max_depth": None, "n_estimators": 200}),
                    ]

classifier_class_l = [(RandomForestClassifier, {"n_estimators": 100})] #, (SVC, {"probability": True, "gamma":"scale"})] #[(DecisionTreeClassifier, {"min_samples_leaf": 30}), (MLPClassifier, {"hidden_layer_sizes": [30,40], "max_iter": 500})]

#a = classifier_class(**classifier_args)

def deviation_off_signed(gtpredictions, predictions):
    """ Compute deviation from optional feature fairness """
    #print("Deviation", gtpredictions.shape, predictions.shape, np.mean(np.abs(predictions-gtpredictions)))
    return np.mean(predictions-gtpredictions)

def deviation_off_sq(gtpredictions, predictions):
    """ Compute squared deviation from optional feature fairness """
    #print("Deviation Sq", gtpredictions.shape, predictions.shape, np.mean((predictions-gtpredictions)**2))
    dev_sq = np.mean((predictions-gtpredictions)**2)
    print(dev_sq)
    return dev_sq

# List of scoring functions
scoring_fns_off = {"off_gap_sq": deviation_off_sq, "off_gap_signed": deviation_off_signed} # Scoring function using the OFF probability as input

if __name__ == "__main__":
    for classifier_class, classifier_args in classifier_class_l:
        print("Using:", classifier_class.__name__, "as estimator.")
        results_dict = {}
        results_dict["plain"] = {}
        #results_dict["offlr"] = {}
        #results_dict["multimodel"] = {}
        results_dict["resampling"] = {}
        results_dict["resampling_same_sz"] = {}
        #results_dict["base"] = {}
        #results_dict["estim_variance"] = {}
        #results_dict["estim_error_plain"] = {}

        # datasets are specified as tuples of samplefn(n_samples), off_proba(X, M), unfair_proba(X, M)
        for dsetname, samplefn, off_proba_gt, unfair_proba_gt in [process_2]: 
            print("Using dataset ", dsetname)
            for n_samples in n_samples_counts:
                results_dict["plain"][n_samples] = {}
                results_dict["resampling"][n_samples] = {}
                results_dict["resampling_same_sz"][n_samples] = {}

                for i in range(n_reps):
                    print("Generating datasets and fitting models ...")
                    print("Using", n_samples, "samples.")
                    # generate train and test sets
                    X, Xext, M, Y  = samplefn(n_samples)

                    #sample a test_set
                    Xt, Xextt, Mtest, yt = samplefn(n_test)
                    full_data = np.hstack((X, Xext))
                    full_data_test = np.hstack((Xt, Xextt))

                    # Fit the models.
                    unfairmodel = classifier_class(**classifier_args)
                    unfairmodel.fit(np.hstack((full_data, M)), Y)

                    resamplemodel = classifier_class(**classifier_args)
                    Xresample, Mresample, Yresample = exhaustive_resampling(full_data, M, Y)
                    resamplemodel.fit(np.hstack((Xresample, Mresample)), Yresample)

                    # Same sample size resample
                    resamplemodel_ss = classifier_class(**classifier_args)
                    Xresamples, Mresamples, Yresamples = random_resampling(full_data, M, Y, n=n_samples)
                    resamplemodel_ss.fit(np.hstack((Xresamples, Mresamples)), Yresamples)

                    gtpred = off_proba_gt(full_data_test, Mtest)[:,1]
                    gt_pred_unfair = unfair_proba_gt(full_data_test, Mtest)[:,1]
                    #print(gtpred[:10], gt_pred_unfair[:10])
                    #exit(0)
                    print("Running evaluation.")
                    #print("Dev unfair", deviation_off(unfairmodel.predict_proba(np.hstack((full_data_test, Mtest))), gt_pred))
                    unfair_predm = unfairmodel.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]
                    evaluate_and_append_scores(scoring_fns_off, results_dict["plain"][n_samples], gtpred, unfair_predm)
                    #evaluate_and_append_scores(scoring_fns_off, results_dict["estim_variance"][n_samples], variance_bl, unfair_predm)
                    #evaluate_and_append_scores(scoring_fns_off, results_dict["estim_error_plain"][n_samples], gt_pred_unfair, unfair_predm)

                    resamplepred = resamplemodel.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]
                    evaluate_and_append_scores(scoring_fns_off, results_dict["resampling"][n_samples], gtpred, resamplepred)
                    if i > 0:
                        results_dict["resampling"][n_samples]["true_sz"].append(len(Xresample))
                    else:
                        results_dict["resampling"][n_samples]["true_sz"] = [len(Xresample)]
                    resamplepred_ss = resamplemodel_ss.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]
                    evaluate_and_append_scores(scoring_fns_off, results_dict["resampling_same_sz"][n_samples], gtpred, resamplepred_ss)

                    

                results_dict["plain"][n_samples] = aggregate_scores(scoring_fns_off, results_dict["plain"][n_samples])
                results_dict["resampling"][n_samples] = aggregate_scores(dict(true_sz=None, **scoring_fns_off), results_dict["resampling"][n_samples])
                results_dict["resampling_same_sz"][n_samples] = aggregate_scores(scoring_fns_off, results_dict["resampling_same_sz"][n_samples])
                

            json.dump(results_dict, open(f"results/exp4/experiment4synth{dsetname}_{classifier_class.__name__}.json","w"))