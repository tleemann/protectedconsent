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

import json

# Set parameters of the data generating processes

## Process 1: Continuous Logistic Process
base_weight = np.array([-1.5,1.0])
base_offset = 0.0

U = np.array([[0.8, 0, 0],[0.4, 0.0, 0]]) # Missingness distribution.
labd = np.array([0.7, 1.0, 0.0])
V = np.array([[0.0, 0, 0.2],[1, -0.15, 0.1]]) # Feature value distribution
mu  = np.array([[-0.25, 0.4, -0.2],[0.25, -0.4, 0.2]]) 

overall_pres = np.array([0.8, 0.3, 0.5])

n_samples_counts = [10000, 50000]

def samplefn_contlog(n_samples):
    Xt, yt = sample_continuous_dataset(base_weight, base_offset, n_features=2, n_records = n_samples)
    yt = yt.astype(int)
    Xextt, Mtest = sample_missing_features(Xt, yt, 3, overall_pres, U, V, labd, mu)
    return Xt, Xextt, Mtest, yt

### Initiate ground truth model parameters according to theory (OFF Model)
gtmodel = ContinousLogisticRegression(n_basefeatures=2, n_missingfeatures=3)
gtmodel.w = base_weight.reshape(1, -1)
gtmodel.t = np.array([base_offset])
gtmodel.omega = U-((mu[1, :] - mu[0, :]).reshape(1,-1)*V)
gtmodel.s = labd + 0.5*(mu[0,:]*mu[0,:] - mu[1,:]*mu[1,:])
gtmodel.beta = mu[1, :] - mu[0, :]

# Non OFF Model
gt_model_unfair = NonFairContinousLogisticRegression(gtmodel, U, labd, overall_pres)

process_1 = ("logistic", samplefn_contlog, gtmodel.predict_proba, gt_model_unfair.predict_proba)

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
n_reps = 6

classifier_class_l = [(RandomForestClassifier, {"n_estimators": 100}),
                    (LogisticRegression, {"solver": "lbfgs"}),
                    (ExtraTreesClassifier, {"n_estimators": 100, "min_samples_split": 10}),
                    (DecisionTreeClassifier, {"min_samples_split": 10}),
                    (GradientBoostingClassifier, {"min_samples_split": 10}),
                    (MLPClassifier, {"hidden_layer_sizes": [30,40], "max_iter": 500}),
                    (SVC, {"probability": True, "gamma":"scale"})
                    ]


classifier_class_l = [(RandomForestClassifier, {})]
#a = classifier_class(**classifier_args)


def accuracy_fn(ys, y_cont):
    return accuracy_score(y_cont > 0.5, ys)

def deviation_off(gtpredictions, predictions):
    """ Compute deviation from optional feature fairness """
    # print("Deviation", gtpredictions.shape, predictions.shape, np.mean(np.abs(predictions-gtpredictions)))
    return np.mean(np.abs(predictions-gtpredictions))

def deviation_off_signed(gtpredictions, predictions):
    """ Compute deviation from optional feature fairness """
    #print("Deviation", gtpredictions.shape, predictions.shape, np.mean(np.abs(predictions-gtpredictions)))
    return np.mean(predictions-gtpredictions)

def deviation_off_sq(gtpredictions, predictions):
    """ Compute squared deviation from optional feature fairness """
    #print("Deviation Sq", gtpredictions.shape, predictions.shape, np.mean((predictions-gtpredictions)**2))
    return np.mean((predictions-gtpredictions)**2)

# List of scoring functions
scoring_fns_label = {"accuracy": accuracy_fn, "rocauc": roc_auc_score} # Scoring function with gt label
scoring_fns_off = {"off_gap": deviation_off, "off_gap_sq": deviation_off_sq, "off_gap_signed": deviation_off_signed} # Scoring function using the OFF probability as input
scoring_fns_all = {**scoring_fns_label, **scoring_fns_off} # merge both


def deviation_off(predictions, gtpredictions):
    """ Compute deviation from optional feature fairness """
    return np.mean(np.abs(predictions[:,1]-gtpredictions[:,1]))

if __name__ == "__main__":
    for classifier_class, classifier_args in classifier_class_l:
        print("Using:", classifier_class.__name__, "as estimator.")
        results_dict = {}
        results_dict["plain"] = {}
        results_dict["offlr"] = {}
        results_dict["multimodel"] = {}
        results_dict["resampling"] = {}
        results_dict["resampling_same_sz"] = {}
        results_dict["base"] = {}
        results_dict["estim_variance"] = {}
        results_dict["estim_error_plain"] = {}

        # datasets are specified as tuples of samplefn(n_samples), off_proba(X, M), unfair_proba(X, M)
        synthetic_dataset_ = [] 
        for dsetname, samplefn, off_proba_gt, unfair_proba_gt in [process_1]:
            print("Using dataset ", dsetname)
            for n_samples in n_samples_counts:
                results_dict["plain"][n_samples] = {}
                results_dict["offlr"][n_samples] = {}
                results_dict["multimodel"][n_samples] = {}
                results_dict["resampling"][n_samples] = {}
                results_dict["base"][n_samples] = {}
                results_dict["estim_variance"][n_samples] = {}
                results_dict["resampling_same_sz"][n_samples] = {}
                results_dict["estim_error_plain"][n_samples] = {}

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
                    # Fit a reference model to estimate variance.
                    var_base_pred = classifier_class(**classifier_args)
                    var_base_pred.fit(np.hstack((full_data, M)), Y)
                    variance_bl = var_base_pred.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]

                    multimodel = FairMultiModel(n_basefeatures=X.shape[1], n_missingfeatures=Xext.shape[1], base_model_cls = classifier_class, **classifier_args)
                    multimodel.fit(full_data, M, Y)

                    fairmodel = ContinousLogisticRegression(n_basefeatures=X.shape[1], n_missingfeatures=Xext.shape[1])
                    fairmodel.fit(full_data, M, Y)

                    unfairmodel = classifier_class(**classifier_args)
                    unfairmodel.fit(np.hstack((full_data, M)), Y)
                    
                    basemodel = classifier_class(**classifier_args)
                    basemodel.fit(X, Y)

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
                    evaluate_and_append_scores(scoring_fns_label, results_dict["plain"][n_samples], yt, unfair_predm)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["plain"][n_samples], gtpred, unfair_predm)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["estim_variance"][n_samples], variance_bl, unfair_predm)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["estim_error_plain"][n_samples], gt_pred_unfair, unfair_predm)

                    offlrpred = fairmodel.predict_proba(full_data_test, Mtest)[:,1]
                    evaluate_and_append_scores(scoring_fns_label, results_dict["offlr"][n_samples], yt, unfair_predm)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["offlr"][n_samples], gtpred, unfair_predm)

                    #print("Dev multimodel", deviation_off()
                    mmpred = multimodel.predict_proba(full_data_test, Mtest)[:,1]
                    evaluate_and_append_scores(scoring_fns_label, results_dict["multimodel"][n_samples], yt, mmpred)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["multimodel"][n_samples], gtpred, mmpred)

                    resamplepred = resamplemodel.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]
                    evaluate_and_append_scores(scoring_fns_label, results_dict["resampling"][n_samples], yt, resamplepred)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["resampling"][n_samples], gtpred, resamplepred)

                    resamplepred_ss = resamplemodel_ss.predict_proba(np.hstack((full_data_test, Mtest)))[:,1]
                    evaluate_and_append_scores(scoring_fns_label, results_dict["resampling_same_sz"][n_samples], yt, resamplepred_ss)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["resampling_same_sz"][n_samples], gtpred, resamplepred_ss)

                    basepred = basemodel.predict_proba(Xt)[:,1]
                    evaluate_and_append_scores(scoring_fns_label, results_dict["base"][n_samples], yt, basepred)
                    evaluate_and_append_scores(scoring_fns_off, results_dict["base"][n_samples], gtpred, basepred)

                    

                results_dict["offlr"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["offlr"][n_samples])
                results_dict["multimodel"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["multimodel"][n_samples])
                results_dict["plain"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["plain"][n_samples])
                results_dict["resampling"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["resampling"][n_samples])
                results_dict["resampling_same_sz"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["resampling_same_sz"][n_samples])
                results_dict["base"][n_samples] = aggregate_scores(scoring_fns_all, results_dict["base"][n_samples])
                results_dict["estim_variance"][n_samples] = aggregate_scores(scoring_fns_off, results_dict["estim_variance"][n_samples])
                results_dict["estim_error_plain"][n_samples] = aggregate_scores(scoring_fns_off, results_dict["estim_error_plain"][n_samples])
                

            json.dump(results_dict, open(f"results/experiment3synth{dsetname}_{classifier_class.__name__}.json","w"))
            #json.dump(results_dict, open(f"results/experiment3synth_{classifier_class.__name__}.json","w"))