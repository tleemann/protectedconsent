## Run the experiment 3 (OFF-LR) with real data.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

from models.multi_model import FairMultiModel
from models.fairmiss_regression import ContinousLogisticRegression
from utils.synthetic_data import create_multiple_sigmoidal_missingness
from utils.data_utils import load_data, budapest_add_missingness_all_columns, evaluate_and_append_scores, aggregate_scores, smart_update_json
from utils.pucida import exhaustive_resampling, random_resampling
import json
# Experimental parameters: Tuple of features and direction for sigmoidal missingness: 1.0 mean 
experiment_params = {
    "compas": (["priors_count", "age", "c_days_from_compas", "c_charge_degree", "juv_misd_count"], [1.0, -1.0, -1.0, 1.0, 1.0]),
    "diabetes": (["Glucose", "Age"], [1.0, 1.0]),
    "adult": (["age", "educational-num", "hours-per-week", "capital-gain", "capital-loss"], [-1.0, -1.0, -1.0, -1.0, -1.0]),
    "californiahousing": (["housing_median_age", "population", "households", "median_income"], [1.0, -1.0, -1.0, -1.0]),
    "income": (["AGEP", "SCHL", "WKHP"], [-1.0, -1.0, -1.0]),
    "insurance": (["experience", "kidslt6", "kids618"], [-1.0, -1.0, -1.0])
}

regression_datasets = ["grades", "income", "californiahousing", "budapest_flat", "insurance"]
# Competitors
competitors = {
    "offlr": False,
    "unfair": True,
    "multi": True,
    "resampling": True,
    "resampling_same_sz": True,
    "base": True
}

# Classification model parameters

classifier_class_l = [  (ExtraTreesClassifier, {"min_samples_split": 10}),
                        (DecisionTreeClassifier, {"min_samples_split": 10}),
                        (GradientBoostingClassifier, {"min_samples_split": 10}),
                        (MLPClassifier, {"hidden_layer_sizes": [30,40], "max_iter": 500}),
                        (RandomForestClassifier, {"min_samples_split": 10}),
                        (RandomForestClassifier, {"max_depth": 4}),
                        (RandomForestClassifier, {"n_estimators": 500}),
                     ]
classifier_class_l = [(RandomForestClassifier, {"n_estimators": 100, "min_samples_split": 10})]
                        
regressor_class_l = [
                      (GradientBoostingRegressor, {"min_samples_split": 10}),
                      (ExtraTreesRegressor, {"min_samples_split": 10}),
]
                                            
regressor_class_l = [(RandomForestRegressor, {"n_estimators": 100, "min_samples_split": 10})]

def accuracy_fn(ys, y_cont):
    return accuracy_score(y_cont > 0.5, ys)

def deviation_off(gtpredictions, predictions):
    """ Compute deviation from optional feature fairness """
    print("Deviation", gtpredictions.shape, predictions.shape, np.mean(np.abs(predictions-gtpredictions)))
    return np.mean(np.abs(predictions-gtpredictions))

def deviation_off_sq(gtpredictions, predictions):
    """ Compute squared deviation from optional feature fairness """
    print("Deviation Sq", gtpredictions.shape, predictions.shape, np.mean((predictions-gtpredictions)**2))
    return np.mean((predictions-gtpredictions)**2)

def mse_times_100(labels, pred):
    return 100 * mean_squared_error(labels, pred)

def mae(labels, pred):
    return 100 * np.mean(np.abs(labels-pred))

def _do_prediction(model, dset, *inputs):
    if dset in regression_datasets:
        return model.predict(*inputs)
    else:
        return model.predict_proba(*inputs)[:,1]
    
# List of scoring functions
scoring_fns_label = {"accuracy": accuracy_fn, "rocauc": roc_auc_score} # Scoring function with gt label
scoring_fn_label_regression = {"mse": mse_times_100, "mae": mae}
scoring_fns_off = {"off_gap": deviation_off, "off_gap_sq": deviation_off_sq} # Scoring function using the OFF probability as input

n_runs = 5
alpha = 1
n_gt_avg = 2 # Number of multimodels to average 

dset_list = ["insurance"] # ["compas, diabetes, adult"] #["insurance"] # ["income", "lawdata", "californiahousing"]
for dset in dset_list: # ["compas", "diabetes", D, "heloc", "adult"]: # experiment_params.keys():
    results_dict = {}
    for item in competitors.keys():
        results_dict[item] = {}

    if dset in regression_datasets:
        model_class_l = regressor_class_l
        scoring_fns_target = scoring_fn_label_regression
    else:
        model_class_l = classifier_class_l
        scoring_fns_target = scoring_fns_label
    scoring_fns_all = {**scoring_fns_target, **scoring_fns_off} # merge both
    for classifier_class, classifier_args in model_class_l:
        class_key = str(classifier_class(**classifier_args))
        print("Using model: ", class_key)
        score_lists_multi = {} # Multimodel
        score_lists_base = {} # Base features only
        score_lists_full = {} # Full feature model
        score_lists_full_multi = {} # Full model implemented through multi models.
        score_lists_offlr = {} # OFF Logistic Regression
        score_lists_exresamplimg = {} # Exhaustive Resampling
        score_lists_resamplimg_same = {} # Same size Resampling

        for i in range(n_runs):
            featurelist, directions = experiment_params[dset]
            n_missing = len(featurelist)
            print(dset, n_missing)
            xs, ys = load_data(dset, normalize_labels=False)

            if dset != "budapest_flat":
                # Compute alphas and introduce missingness
                feature_stds = xs[featurelist].std()
                alphas_missingness = np.array([alpha*sign/feature_stds[fname] for fname, sign in zip(featurelist, directions)])
                xs_missing = create_multiple_sigmoidal_missingness(xs, featurelist, alphas_missingness)
            else:
                xs_missing = budapest_add_missingness_all_columns(xs, featurelist)
            xtrain, xtest, ytrain, ytest = train_test_split(xs_missing, ys, test_size=0.2, train_size=0.8, random_state=42*i, shuffle=True)

            to_drop = [mfeature + " missing" for mfeature in featurelist]
            avail_train= 1-xtrain[to_drop]
            xtrain = xtrain.drop(to_drop, axis=1)
            avail_test= 1-xtest[to_drop]
            xtest = xtest.drop(to_drop, axis=1)
            
            # Train the fair model.
            gt_pred_list = []
            for i in range(n_gt_avg):
                gtmodel = FairMultiModel(n_basefeatures = xtrain.shape[1]-n_missing, n_missingfeatures=n_missing, 
                        base_model_cls=classifier_class, random_state=42*i, **classifier_args)
                gtmodel.fit(xtrain.values, 1-avail_train.values, ytrain.values)
                gt_pred_list.append(_do_prediction(gtmodel, dset, xtest.values, 1-avail_test.values)) #gtmodel.predict_proba(xtest.values, missing_test.values)[:,1])
            gt_pred_list_stack=np.stack(gt_pred_list)
            gtpred = gt_pred_list_stack.mean(axis=0)
            evaluate_and_append_scores(scoring_fns_target, score_lists_multi, ytest.values, gtpred)
            evaluate_and_append_scores(scoring_fns_off, score_lists_multi, gt_pred_list_stack[0,:], gtpred)

            if competitors["offlr"]:
                # Train the OFF Logistic Regression model
                logregoffmodel = ContinousLogisticRegression(n_basefeatures = xtrain.shape[1]-n_missing, n_missingfeatures=n_missing, random_state=42*i)
                logregoffmodel.fit(xtrain.values.astype(float), 1-avail_train.values, ytrain.values)
                logregoffmodelpred = _do_prediction(logregoffmodel, dset, xtest.values, 1-avail_test.values) #logregoffmodel.predict_proba(xtest.values, missing_test.values)[:,1]
                evaluate_and_append_scores(scoring_fns_target, score_lists_offlr, ytest.values, logregoffmodelpred)
                evaluate_and_append_scores(scoring_fns_off, score_lists_offlr, gtpred, logregoffmodelpred)

            if competitors["unfair"]:
                #Train the unfair model
                unfair_model = classifier_class(random_state=42*i, **classifier_args)
                unfair_model.fit(np.hstack((xtrain.values, avail_train.values)), ytrain.values)
                unfair_pred = _do_prediction(unfair_model, dset, np.hstack((xtest.values, avail_test.values))) #unfair_model.predict_proba(np.hstack((xtest.values, missing_test.values)))[:,1]
                evaluate_and_append_scores(scoring_fns_target, score_lists_full, ytest.values, unfair_pred)
                evaluate_and_append_scores(scoring_fns_off, score_lists_full, gtpred, unfair_pred)

            if competitors["resampling"]:
                #Train the resampling model
                Xres, Mres, Yres = exhaustive_resampling(xtrain.values.astype(float), avail_train.values, ytrain.values)
                resample_model = classifier_class(random_state=42*i, **classifier_args)
                resample_model.fit(np.hstack((Xres, Mres)), Yres)
                resample_pred = _do_prediction(resample_model, dset, np.hstack((xtest.values.astype(float), avail_test.values)))
                evaluate_and_append_scores(scoring_fns_target, score_lists_exresamplimg, ytest.values, resample_pred)
                evaluate_and_append_scores(scoring_fns_off, score_lists_exresamplimg, gtpred, resample_pred)
                ## Append the sample sz.
                if "true_sz" in score_lists_exresamplimg.keys():
                    score_lists_exresamplimg["true_sz"].append(len(Xres))
                else:
                    score_lists_exresamplimg["true_sz"] = [len(Xres)]

            if competitors["resampling_same_sz"]:
                #Train the resampling model
                Xres, Mres, Yres = random_resampling(xtrain.values.astype(float), avail_train.values, ytrain.values, n=len(xtrain))
                resample_model = classifier_class(random_state=42*i, **classifier_args)
                resample_model.fit(np.hstack((Xres, Mres)), Yres)
                resample_pred_same = _do_prediction(resample_model, dset, np.hstack((xtest.values.astype(float), avail_test.values)))
                evaluate_and_append_scores(scoring_fns_target, score_lists_resamplimg_same, ytest.values, resample_pred_same)
                evaluate_and_append_scores(scoring_fns_off, score_lists_resamplimg_same, gtpred, resample_pred_same)
                if "true_sz" in score_lists_resamplimg_same.keys():
                    score_lists_resamplimg_same["true_sz"].append(len(xtrain))
                else:
                    score_lists_resamplimg_same["true_sz"]=[len(xtrain)]

            if competitors["base"]:
                #Train the base model
                base_logreg = classifier_class(random_state=42*i, **classifier_args)
                xtrain_base = xtrain.drop(featurelist, axis=1)
                xtest_base = xtest.drop(featurelist, axis=1)
                base_logreg.fit(xtrain_base.values, ytrain.values)
                base_pred = _do_prediction(base_logreg, dset, xtest_base.values)
                evaluate_and_append_scores(scoring_fns_target, score_lists_base, ytest.values, base_pred)
                evaluate_and_append_scores(scoring_fns_off, score_lists_base, gtpred, base_pred)
        
        # Aggregate results over runs
        results_dict["multi"][class_key] = aggregate_scores(scoring_fns_all, score_lists_multi)
        if competitors["base"]:
            results_dict["base"][class_key] = aggregate_scores(scoring_fns_all, score_lists_base)
        if competitors["unfair"]:
            results_dict["unfair"][class_key] = aggregate_scores(scoring_fns_all, score_lists_full)
        if competitors["offlr"]:
            results_dict["offlr"][class_key] = aggregate_scores(scoring_fns_all, score_lists_offlr)
        if competitors["resampling"]:
            results_dict["resampling"][class_key] = aggregate_scores(dict(true_sz=0, **scoring_fns_all), score_lists_exresamplimg)
        if competitors["resampling_same_sz"]:
            results_dict["resampling_same_sz"][class_key] = aggregate_scores(dict(true_sz=0, **scoring_fns_all), score_lists_resamplimg_same)
    smart_update_json(results_dict, f"results/exp3/exp3real_{dset}.json")





