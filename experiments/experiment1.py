import numpy as np
import pandas as pd
import sklearn
import openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split
from utils.pucida import exhaustive_resampling
import utils.data_utils as utils
from utils.data_utils import evaluate_and_append_scores, evaluate_and_append_scores_ma, aggregate_scores, smart_update_json
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
# reproducibility
np.random.seed(45)

# Set parameters for the classification data
classification_datasets = ["diabetes", "compas", "adult"]

voluntary_features_cls = {"diabetes": "Glucose", "compas": "priors_count", "adult": "educational-num"}
alpha_vals_cls = {"compas": 1.0, "diabetes": 0.05, "adult": -0.8}
positive_outcome = {"compas": 0, "diabetes": 0, "adult": 1}


classifier_class_l = [  (ExtraTreesClassifier, {"min_samples_split": 10}),
                        (DecisionTreeClassifier, {"min_samples_split": 10}),
                        (GradientBoostingClassifier, {"min_samples_split": 10}),
                        (MLPClassifier, {"hidden_layer_sizes": [30,40], "max_iter": 500}),
                        (RandomForestClassifier, {"min_samples_split": 10}),
                        (RandomForestClassifier, {"max_depth": 4}),
                        (RandomForestClassifier, {"n_estimators": 500}),
                    ]
classifier_class_l = [(RandomForestClassifier, {"max_depth": None})]

binary_pred = True
adversarial_missingness = False
use_imputation = "zero"

# Set parameters for the regression data
regression_datasets = ["income", "californiahousing",  "insurance"]
voluntary_features_reg = {"californiahousing": "median_income", "income": "WKHP", "insurance": "experience"}
alpha_vals_reg = {"californiahousing": -2.0, "income": -0.2, "insurance": 11}


regressor_class_l = [  (ExtraTreesRegressor, {"min_samples_split": 10}),
                        (DecisionTreeRegressor, {"min_samples_split": 10}),
                        (GradientBoostingRegressor, {"min_samples_split": 10}),
                        (RandomForestRegressor, {"min_samples_split": 10}),
                        (RandomForestRegressor, {"max_depth": 4}),
                        (RandomForestRegressor, {"n_estimators": 500}),
                    ]

regressor_class_l = [(RandomForestRegressor,  {"max_depth": None})] 

# Set general parameters
n_reps = 5
cv_param = 5
# Scoring functions
def id(labels, preds):
    return preds

def mean_pred(labels, preds):
    #print(preds)
    if binary_pred and not regression_flag:
        return np.mean(preds > 0.5)
    else:
        return np.mean(preds)

def abs_gap(missing_pred_base, missing_pred_full): #find a way to deal with score = score_fn(labels, preds)
    missing_pred_base = np.mean(missing_pred_base > 0.5) if (binary_pred and not regression_flag) else np.mean(missing_pred_base)
    missing_pred_full = np.mean(missing_pred_full > 0.5) if (binary_pred and not regression_flag) else np.mean(missing_pred_full)
    return missing_pred_base - missing_pred_full

def abs_gap_reg(missing_pred_base, missing_pred_full):
    missing_pred_base = np.mean(missing_pred_base)
    missing_pred_full = np.mean(missing_pred_full)
    return missing_pred_base - missing_pred_full

def rel_gap_reg(pred_base, pred_mis):
    #rel_gap = utils.avg_diff_per_datapoint(missing_pred_base, missing_pred_full, dataset)
    avg_percentage_change = (np.mean(pred_mis) - np.mean(pred_base)) / np.absolute(np.mean(pred_base))
    return avg_percentage_change

def mse_avail(labels, avail_preds):
    return utils.mse_loss(avail_preds, labels)

def accuracy_fn(ys, y_cont):
    return 100 * (accuracy_score(y_cont > 0.5, ys))

def cost(labels, preds):
    if np.var(labels) == 0:
        return 100.0
    else:
        return 100 * (roc_auc_score(labels, preds))

def _do_prediction(model, dset, *inputs):
    if dset in regression_datasets:
        return model.predict(*inputs)
    else:
        return model.predict_proba(*inputs)[:,1]
    
# List of scoring functions
scoring_fns_pred = {"pred": mean_pred} #Scoring function without labels, only looks at avg predictions based on datapoints with voluntary feature missing
scoring_fns_cls_abs_gap = {"abs_gap": abs_gap} # Scoring function with predictions of base model and full/resampling model
scoring_fns_reg_abs_gap = {"abs_gap": abs_gap_reg}
scoring_fns_reg_rel_gap = {"rel_gap": rel_gap_reg} # Scoring function that needs to undo data transformation before scoring
scoring_fns_reg_mse = {"mse": mse_avail} # Scoring function needs gt labels
scoring_fns_cls_costs = {"acc": accuracy_fn, "roc": cost}
scoring_fns_reg_all = {**scoring_fns_pred, **scoring_fns_reg_abs_gap, **scoring_fns_reg_rel_gap,  **scoring_fns_reg_mse} # Merge all scoring functions for convenient saving
scoring_fns_cls_all = {**scoring_fns_pred, **scoring_fns_cls_abs_gap}
scoring_fns_reg_base = {**scoring_fns_pred, **scoring_fns_reg_mse}

# Competing models
competitors = {
    "base": True,
    "unfair": True,
    "resampling": True
}

dset_list = ["income", "insurance", "californiahousing", "diabetes", "compas", "adult", "25,pathology_cp_data,abdominocentesis_appearance", "940,binaryClass,RD-DBO-P"]

if __name__ == "__main__":
    for dset in dset_list:
        results_dict = {}
        for item in competitors.keys():
            results_dict[item] = {}
        if dset in (regression_datasets+classification_datasets):
            if dset in regression_datasets:
                model_class_l = regressor_class_l
                regression_flag = True
                voluntary_feature_dict = voluntary_features_reg
                alpha_vals = alpha_vals_reg
            elif dset in classification_datasets:
                model_class_l = classifier_class_l
                regression_flag = False
                voluntary_feature_dict = voluntary_features_cls
                alpha_vals = alpha_vals_cls
            col_name = voluntary_feature_dict[dset]
            print(dset, col_name)
            xs, ys = utils.load_data(dset)
            xs_base = xs.drop(col_name, axis=1)
            a, b = [alpha_vals[dset], xs[col_name].mean()]

        elif "," in dset: # in real_missing_datasets:
            model_class_l = classifier_class_l
            regression_flag = False
            voluntary_feature_dict = voluntary_features_cls
            alpha_vals = alpha_vals_cls
            parts = dset.split(",")
            data_id = int(parts[0])
            target_feature = parts[1]
            dataset = openml.datasets.get_dataset(data_id)
            print(data_id)
            dataframe, ys, _, _ = dataset.get_data(target=target_feature, dataset_format="dataframe")
            col_name = parts[2]

        for classifier_class, classifier_args in model_class_l:
            class_key = str(classifier_class(**classifier_args))
            print("Using model: ", class_key)
            score_lists_base = {} #Base features only
            score_lists_full = {} #Full feature model
            score_lists_resampling = {} #Resampling model

            cm_dict_full = {"full": np.zeros([2,2]), "subm": np.zeros([2,2]), "suba": np.zeros([2,2])}
            cm_dict_orig = {"full": np.zeros([2,2])}
            dflip_dict = {"full": np.zeros(2), "subm": np.zeros(2), "suba": np.zeros(2)}
            for i in range(n_reps):
                model = classifier_class(random_state=42*i, **classifier_args)
                if not adversarial_missingness:
                    xs_mis = utils.create_missingness(xs, col_name, a, b, verbose=False, imputation_val=use_imputation)
                else:
                    xs_mis = utils.create_adversarial_missingness(xs, col_name, ys, classifier_class, regression_flag=regression_flag, imputation_val=use_imputation)

                A = 1 - xs_mis[col_name + " missing"].values
                X = pd.concat((xs_base, xs_mis[col_name]), axis=1) # make sure the clm in question is last
                Xr, Ar, Yr, org_indicator = exhaustive_resampling(X.values, A.reshape(-1,1), ys.values, return_original_indictor=True)

                # Training
                if regression_flag:
                    pred_base_model = cross_val_predict(model, xs_base.values, ys.values, cv=cv_param) #Base model
                    pred_full_model = cross_val_predict(model, xs_mis.values, ys.values, cv=cv_param) #Full model
                    pred_resampling_model = cross_val_predict(model, np.hstack((Xr, Ar)), Yr, cv=cv_param)[org_indicator==1] #Resampling model
                else:
                    pred_base_model = cross_val_predict(model, xs_base.values, ys.values, cv=cv_param, method ="predict_proba")[:,1]
                    pred_full_model = cross_val_predict(model, xs_mis.values, ys.values, cv=cv_param, method ="predict_proba")[:,1]
                    pred_resampling_model = cross_val_predict(model, np.hstack((Xr, Ar)), Yr, cv=cv_param, method ="predict_proba")[:,1][org_indicator==1]
                    pred_orig_model = cross_val_predict(model, xs.values, ys.values, cv=cv_param, method ="predict_proba")[:,1] # Model trained on original data (without added missingness)

                if adversarial_missingness:
                    print("Updating", np.sum(pred_base_model > pred_resampling_model), "predictions to obtain bonus system.")
                    pred_resampling_model = np.max(np.stack((pred_base_model, pred_resampling_model)), axis=0)
                    print(pred_resampling_model.shape)
                idx_mis = xs_mis[col_name + " missing"][xs_mis[col_name + " missing"]==1].index
                idx_avail = xs_mis[col_name + " missing"][xs_mis[col_name + " missing"]==0].index
                labels_miss = ys[idx_mis.values.flatten()]
                labels_avail = ys[idx_avail.values.flatten()]
                missing_pred_base = pd.DataFrame(pd.DataFrame(pred_base_model), index=idx_mis).values
                avail_pred_base = pd.DataFrame(pd.DataFrame(pred_base_model), index=idx_avail).values
                missing_pred_full =  pd.DataFrame(pd.DataFrame(pred_full_model), index=idx_mis).values
                avail_pred_full = pd.DataFrame(pd.DataFrame(pred_full_model), index=idx_avail).values                
                missing_pred_resampling = pred_resampling_model[idx_mis.values.flatten()]
                avail_pred_resampling = pred_resampling_model[idx_avail.values.flatten()]

                # Scoring
                # Get predictions for all models
                evaluate_and_append_scores_ma(scoring_fns_pred, score_lists_base, None, missing_pred_base, None, avail_pred_base) # these scores don't need labels, so we give None
                evaluate_and_append_scores_ma(scoring_fns_pred, score_lists_full, None, missing_pred_full, None, avail_pred_full)
                evaluate_and_append_scores_ma(scoring_fns_pred, score_lists_resampling, None, missing_pred_resampling, None, avail_pred_resampling)
                if regression_flag:
                    # Get absolute prediction gaps between base model and full/resampling model
                    evaluate_and_append_scores(scoring_fns_reg_abs_gap, score_lists_full, missing_pred_base, missing_pred_full)
                    evaluate_and_append_scores(scoring_fns_reg_abs_gap, score_lists_resampling, missing_pred_base, missing_pred_resampling)
                    # Get relative prediction gaps between base model and full/resampling model
                    evaluate_and_append_scores(scoring_fns_reg_rel_gap, score_lists_full, utils.undo_transformation(missing_pred_base, dset), utils.undo_transformation(missing_pred_full, dset))
                    evaluate_and_append_scores(scoring_fns_reg_rel_gap, score_lists_resampling, utils.undo_transformation(missing_pred_base, dset), utils.undo_transformation(missing_pred_resampling, dset))
                    # Get mse for predictions on datapoints with voluntary feature available
                    #mse_avail(avail_pred_base.flatten(), ys[idx_avail.values.flatten()])
                    #evaluate_and_append_scores(scoring_fns_cls_costs, score_lists_base, ys[idx_avail.values.flatten()], avail_pred_base.flatten())
                    #evaluate_and_append_scores(scoring_fns_cls_costs, score_lists_full, ys[idx_avail.values.flatten()], avail_pred_full.flatten())
                    #evaluate_and_append_scores(scoring_fns_cls_costs, score_lists_resampling, ys[idx_avail.values.flatten()], avail_pred_resampling.flatten())
                    evaluate_and_append_scores_ma(scoring_fns_reg_mse, score_lists_base, labels_miss, missing_pred_base.flatten(), labels_avail, avail_pred_base.flatten()) # these scores don't need labels, so we give None
                    evaluate_and_append_scores_ma(scoring_fns_reg_mse, score_lists_full, labels_miss, missing_pred_full.flatten(), labels_avail, avail_pred_full.flatten())
                    evaluate_and_append_scores_ma(scoring_fns_reg_mse, score_lists_resampling, labels_miss, missing_pred_resampling.flatten(), labels_avail, avail_pred_resampling.flatten())
                    
                else:
                    # Get absolute prediction gaps between base model and full/resampling model
                    evaluate_and_append_scores(scoring_fns_cls_abs_gap, score_lists_full, missing_pred_base, missing_pred_full)
                    evaluate_and_append_scores(scoring_fns_cls_abs_gap, score_lists_resampling, missing_pred_base, missing_pred_resampling)
                    evaluate_and_append_scores_ma(scoring_fns_cls_costs, score_lists_base, labels_miss, missing_pred_base, labels_avail, avail_pred_base) # these scores don't need labels, so we give None
                    evaluate_and_append_scores_ma(scoring_fns_cls_costs, score_lists_full, labels_miss, missing_pred_full, labels_avail, avail_pred_full)
                    evaluate_and_append_scores_ma(scoring_fns_cls_costs, score_lists_resampling, labels_miss, missing_pred_resampling, labels_avail, avail_pred_resampling)
                    
                    # For appendix, get some additional scores
                    # Get confusion matrices
                    cm_dict_full["full"] += confusion_matrix(ys.values, utils.proba_to_classification(pred_full_model))
                    cm_dict_full["subm"] += confusion_matrix(ys[idx_mis.values.flatten()].values, utils.proba_to_classification(missing_pred_full)) # full model, subset of missing datapoints
                    cm_dict_full["suba"] += confusion_matrix(ys[idx_avail.values.flatten()].values, utils.proba_to_classification(avail_pred_full)) # full model, subset of available (non-missing) datapoints
                    cm_dict_orig["full"] += confusion_matrix(ys.values, utils.proba_to_classification(pred_orig_model))
                    # Get decision flipping ratios
                    dflip_dict["full"] += utils.get_dflip_ratios(pred_orig_model, pred_full_model)
                    dflip_dict["subm"] += utils.get_dflip_ratios(pred_orig_model[idx_mis], pred_full_model[idx_mis])
                    dflip_dict["suba"] += utils.get_dflip_ratios(pred_orig_model[idx_avail], pred_full_model[idx_avail])

            # Aggregate results over runs
            scoring_fns = scoring_fns_reg_all if regression_flag else scoring_fns_cls_all
            scoring_fns_base = scoring_fns_reg_base if regression_flag else scoring_fns_pred
            results_dict["base"][class_key] = aggregate_scores(scoring_fns_base, score_lists_base)
            results_dict["unfair"][class_key] = aggregate_scores(scoring_fns, score_lists_full)
            results_dict["resampling"][class_key] = aggregate_scores(scoring_fns, score_lists_resampling) 
            if regression_flag==False:
                results_dict["appendix scores"] = { "FPR Full model, full data": utils.get_fpr(cm_dict_full["full"]/n_reps),
                                                    "FPR Full model, missing data": utils. get_fpr(cm_dict_full["subm"]/n_reps),
                                                    "FPR Full model, non-missing data": utils. get_fpr(cm_dict_full["suba"]/n_reps),
                                                    "FPR Full model, original data": utils.get_fpr(cm_dict_orig["full"]/n_reps),
                                                    "FNR Full model, full data": utils.get_fnr(cm_dict_full["full"]/n_reps),
                                                    "FNR Full model, missing data": utils. get_fnr(cm_dict_full["subm"]/n_reps),
                                                    "FNR Full model, non-missing data": utils. get_fnr(cm_dict_full["suba"]/n_reps),
                                                    "FNR Full model, original data": utils.get_fnr(cm_dict_orig["full"]/n_reps),
                                                    "Pos. flip ratio, full data": dflip_dict["full"][0]/n_reps,
                                                    "Pos. flip ratio, missing data": dflip_dict["subm"][0]/n_reps,
                                                    "Pos. flip ratio, non-missing data": dflip_dict["suba"][0]/n_reps,
                                                    "Neg. flip ratio, full data": dflip_dict["full"][1]/n_reps,
                                                    "Neg. flip ratio, missing data": dflip_dict["subm"][1]/n_reps,
                                                    "Neg. flip ratio, non-missing data": dflip_dict["suba"][1]/n_reps
                            }
            results_dict["voluntary feature"] = col_name
            results_dict["model"] = class_key

        mode = "bin" if not adversarial_missingness else "adv"
        if use_imputation != "zero":
            mode = mode+use_imputation

        if binary_pred == True:
            smart_update_json(results_dict, f"results/exp1{mode}/exp1_{dset}.json")
        else:
            smart_update_json(results_dict, f"results/exp1{mode}/exp1_{dset}.json")
