#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
pd.set_option('mode.chained_assignment', None) #prevent unnecessary warning


# normalize data along feature axis (min-max-scaling)
def min_max_scaling(ys):
    return (ys - ys.min(axis=0)) / (ys.max(axis=0) - ys.min(axis=0))


def undo_transformation(preds_scaled, dataset):
    if dataset=="heloc":
        _, ys = load_data(dataset, regression=True)
        preds_unscaled = preds_scaled * (ys.max(axis=0) - ys.min(axis=0)) + ys.min(axis=0)
    elif dataset in ["californiahousing", "budapest_flat_view", "budapest_flat_pet", "income"]:
        preds_unscaled = np.exp(preds_scaled)
    elif dataset=="grades":
        data = pd.read_csv("data/student-por.csv", sep=";")
        m = (data[["G1", "G2", "G3"]].astype(int)).mean(axis=1)
        preds_unscaled = preds_scaled*10.0+ m.mean()
    elif dataset=="lawdata":
        _, ys = load_data(dataset, regression=True)
        preds_unscaled = (preds_scaled - ys.min(axis=0)) / (ys.max(axis=0) - ys.min(axis=0))
        preds_unscaled = preds_unscaled * 4.0 #we don't know the original scale, so we choose the range 1-10
        # TODO check if values are in range 1-10
    else:
        preds_unscaled = preds_scaled
    return preds_unscaled

def sigmoid(x, a, b):
    return 1/(1 + np.exp(-a*(x-b)))


def load_data(data_name, normalize_labels=False, use_dummy_encoding=True, regression=False):
    if data_name=="diabetes":
        data = pd.read_csv("data/diabetes.csv")
        xs = data.drop(["Outcome"], axis=1)
        ys = data["Outcome"]  
    elif data_name=="compas":
        data = pd.read_csv("data/compas_preprocessed.csv")
        xs = data.drop(["two_year_recid", "decile_score", "Unnamed: 0"], axis=1)
        ys = data["two_year_recid"]
    elif data_name=="adult":
        data = pd.read_csv("data/adult.csv")
        xs = data.drop(["income", "fnlwgt", "education"], axis=1)
        ys = data["income"].replace([">50K", "<=50K"], [1, 0])
    elif data_name=="californiahousing":
        data = pd.read_csv("data/housing.csv")
        xs = data.drop(["median_house_value"], axis=1)
        ys = np.log(data["median_house_value"])
    elif data_name=="income":
        data = pd.read_csv("data/income.csv")
        xs = data.drop(["income"], axis=1)
        ys = np.log(data["income"])
    elif data_name=="insurance":
        data = pd.read_csv("data/insurance.csv")
        xs = data.drop(["whrswk"], axis=1)
        ys = data["whrswk"]
    if use_dummy_encoding==True:
        xs = dummy_encoding(xs)
    return xs, ys   


# encode categorical features with dummy features (one-hot encoding)
def dummy_encoding(data):
    for col in list(data.columns.values):
        if type(data[col][0])==str:
            dummies = pd.get_dummies(data[col])
            data = pd.concat([data, dummies], axis=1).reindex(data.index)
            data.drop(col, axis=1, inplace=True)
    
    data.fillna(value=0, inplace=True) 
    return data


def get_model_acc(xs, ys, model=LogisticRegression()):
    cross_val = cross_validate(model, xs.values, ys.values, cv=10, return_estimator=True)
    accuracy = np.mean(cross_val["test_score"])
    return accuracy


def print_model_accs(data_name, model_list):
    print("########## " + data_name + " ##########")
    xs, ys = load_data(data_name, normalize=True)
    for model in model_list:
        print(str(model) + ": " + str(get_model_acc(xs, ys, model)))
    print("Baseline:", max(ys.mean(), 1-ys.mean()))


def acc_per_feature(data_name="german", normalize=False, model= LogisticRegression()):
    xs, ys = load_data(data_name, normalize)
    print("########## " + data_name + " ##########")
    print("Baseline classifier (only positive prediction): " + str(np.mean(ys)))
    for feature in xs.columns:
        cross_val = cross_validate(model, xs[feature].to_frame().values, ys.values, cv=10, return_estimator=True)
        accuracy = np.mean(cross_val["test_score"])
        print(feature + ": " + str(accuracy))
    return


def mse_per_feature(data_name="german", normalize=False, model= LogisticRegression()):
    """ For Regression. """
    xs, ys = load_data(data_name, normalize)
    print("########## " + data_name + " ##########")
    #Naive error:
    nerror = np.mean((ys-ys.mean())**2)
    print("Baseline classifier (only positive prediction): ", nerror)
    for feature in xs.columns:
        cross_val = cross_validate(model, xs[feature].to_frame().values, ys.values, cv=5, scoring=("neg_mean_squared_error",))
        #print(cross_val)
        accuracy = -np.mean(cross_val["test_neg_mean_squared_error"])
        print(feature + ": " + str(accuracy))
    return


def mse_leaveout_feature(data_name="german", normalize=False, model= LogisticRegression()):
    """ For Regression. """
    xs, ys = load_data(data_name, normalize)
    print("########## " + data_name + " ##########")
    #Naive error:
    nerror = np.mean((ys-ys.mean())**2)
    print("Baseline classifier (only positive prediction): ", nerror)
    for feature in xs.columns:
        xs_mod = xs.drop(feature, axis=1)
        cross_val = cross_validate(model, xs_mod.values, ys.values, cv=5, scoring=("neg_mean_squared_error",))
        #print(cross_val)
        accuracy = -np.mean(cross_val["test_neg_mean_squared_error"])
        print(feature + ": " + str(accuracy))
    return
    

# Create binary variable
def create_binary_feature(xs, feature, val, new_feature_name):
    xs_copy = xs.copy(deep=True)
    idx = xs_copy[feature][(xs_copy[feature] == val)].index
    #xs_copy[feature].iloc[idx] = np.random.choice(vals)
    xs_copy[new_feature_name] = 0 #append column of all-zeros
    xs_copy[new_feature_name].iloc[idx] = 1 # set to one for indices where -7 occurs
    return xs_copy[new_feature_name]


def shap_val(data_name, normalize=False, model=LogisticRegression()):
    xs, ys = load_data(data_name, normalize, dummy_encoding=False)
    print("########## " + data_name + " ##########")
    cross_val = cross_validate(model, dummy_encoding(xs).values, ys.values, cv=10, return_estimator=False)
    full_accuracy = np.mean(cross_val["test_score"])
    print("Accuracy for full dataset: " + str(full_accuracy))
    print("Format: column_name: new_accuracy - full_accuracy")
    for feature in xs.columns:
        cross_val = cross_validate(model, dummy_encoding(xs.drop(feature, axis=1)).values, ys.values, cv=10, return_estimator=False)
        accuracy = np.mean(cross_val["test_score"])
        print(feature + ": " + str(accuracy - full_accuracy))
    return


## For now implement only df_data (meaning we use the sample mean as missingness value)
def create_adversarial_missingness(df, col_name, ys, model_class, regression_flag, imputation_val="zero"):
    """ Create missingness by applying a sigmoidal missingness function to each feature.
        p(missing) = sigmoid(a*(x-b)) 
        if a is positive, missingness is more probably for higher values of the feature col_name.
    """
    df_copy = df.copy(deep=True)
    if regression_flag:
        pred_full_model = cross_val_predict(model_class(), df.values, ys.values, cv=2) # Full model
    else:
        pred_full_model = cross_val_predict(model_class(), df.values, ys.values, cv=2, method="predict_proba")[:,1]
    
    xs_base = df.drop(col_name, axis=1)
    if regression_flag:
        pred_base_model = cross_val_predict(model_class(), xs_base.values, ys.values, cv=2) # Full model
    else:
        pred_base_model = cross_val_predict(model_class(), xs_base.values, ys.values, cv=2, method="predict_proba")[:,1]

    # Now drop all predictions, where the score is better for base.
    missing = pred_base_model > pred_full_model
    print("Creating adversarial missingness for", np.sum(missing), "values.")
    # Create dummy columns to encode missingness
    df_copy[col_name + " missing"] = missing 
    voluntary_feature = df_copy[col_name]
    if imputation_val=="zero":
        voluntary_feature.iloc[missing] = 0 # np.mean(df_copy[col_name].values)
    elif imputation_val=="mean":
        voluntary_feature.iloc[missing] = np.mean(df_copy[col_name].values)
    elif imputation_val=="median":
        voluntary_feature.iloc[missing] = np.median(df_copy[col_name].values)
    
    print(df_copy.columns)
    return df_copy



## For now implement only df_data (meaning we use the sample mean as missingness value)
def create_missingness(df, col_name, a, b, verbose=True, imputation_val="zero"):
    """ Create missingness by applying a sigmoidal missingness function to each feature.
        p(missing) = sigmoid(a*(x-b)) 
        if a is positive, missingness is more probably for higher values of the feature col_name.
    """
    df_copy = df.copy(deep=True)
    # Create dummy columns to encode missingness
    df_copy[col_name + " missing"] = df_copy[col_name].apply(lambda x: sigmoid(x,a,b))
    idx = df_copy[col_name + " missing"][np.random.uniform(size=len(df_copy[col_name + " missing"].values)) < 
                                         df_copy[col_name + " missing"].values].index
    voluntary_feature = df_copy[col_name]
    if imputation_val=="zero":
        voluntary_feature.iloc[idx] = 0 # np.mean(df_copy[col_name].values)
    elif imputation_val=="mean":
        voluntary_feature.iloc[idx] = np.mean(df_copy[col_name].values)
    elif imputation_val=="median":
        voluntary_feature.iloc[idx] = np.median(df_copy[col_name].values)
        
    df_copy[col_name] = voluntary_feature
    binary_missingness_feature = df_copy[col_name + " missing"]
    binary_missingness_feature = df_copy[col_name].apply(lambda x: 0)
    binary_missingness_feature.iloc[idx] = 1
    df_copy[col_name + " missing"] = binary_missingness_feature
    
    if verbose:
        print("Voluntary feature:", col_name)
        print("Instances missing:", df_copy[col_name + " missing"].value_counts().sort_index()[0], "out of", 
              len(df_copy[col_name + " missing"].values), ".")
    return df_copy


def budapest_add_missingness_col(df, col_name):
    df_copy = df.copy(deep=True)
    df_copy[col_name + " missing"] = df_copy[col_name].apply(lambda x: 0)
    df_copy[col_name + " missing"][df_copy[col_name]==0] = 1
    return df_copy

def budapest_add_missingness_all_columns(df, col_names):
    df_copy = df.copy(deep=True)
    all_possible_cols = ["pets_allowed", "view_type", "garden"]
    for c in all_possible_cols:
        if c in col_names:
            df_copy[c + " missing"] = df_copy[c].apply(lambda x: 0)
            df_copy[c + " missing"][df_copy[c]==0] = 1
        else:
            df_copy = df_copy.drop(c, axis=1)
    print(df_copy.columns)
    return df_copy

def plot_missing_fun(data_name, col_name, a, b, savepath=None):
    df, _ = load_data(data_name)
    x = np.linspace(min(df[col_name].value_counts().index), max(df[col_name].value_counts().index))
    y_sig = sigmoid(x,a,b)
    
    plt.rc('font', size=15) #increase font size from 10 (default) to 15
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(col_name + " (λ=" + str(a) + ")")
    ax1.set_ylabel("Num. Occurences", color = "black") 
    plot_1 = ax1.hist(df[col_name], 15, density=False, color="black")
    ax1.tick_params(axis ="y", labelcolor = "black") 
    
    #if data_name=="lawdata":
    #    #ax1.set_xlim(right=60)
    #    x = np.linspace(min(df[col_name].value_counts().index), 60)
    #    y_sig = sigmoid(x,a,b)
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel("P(missing)", color = "blue") 
    plot_2 = ax2.plot(x, y_sig, color = "blue") 
    ax2.tick_params(axis ="y", labelcolor = "blue")
    plt.title(data_name)
    
    if savepath != None:
        plt.savefig(savepath, dpi=250, format="png", bbox_inches="tight")
    
def get_gap(xs, xs_mis, ys, col_name, model):
    
    pred_base_model = cross_val_predict(model, xs.values, ys.values, cv=10)
    pred_missing_model = cross_val_predict(model, xs_mis.values, ys.values, cv=10)
    idx_mis = xs_mis[col_name + " missing"][xs_mis[col_name + " missing"]==1].index
    missing_pred_base = pd.DataFrame(pd.DataFrame(pred_base_model), index=idx_mis)
    missing_pred_mis = pd.DataFrame(pd.DataFrame(pred_missing_model), index=idx_mis)
    gap = missing_pred_base.mean().values - missing_pred_mis.mean().values
    """
    print("missing_pred_base.mean(): " + str(missing_pred_base.mean()))
    print("missing_pred_base.mean(): " + str(missing_pred_mis.mean()))
    print("gap: " + str(gap))
    """
    return gap
    
def aggregate_and_store_scores(score_lists, dataset, model_type, classifier_class, voluntary_feature_name, regression=False):
    results_dict = {}
    results_dict[model_type] = {"voluntary feature": voluntary_feature_name,
                                "classifier class": str(classifier_class)
                                }
    for score_name, score_list in score_lists.items():
        results_dict[model_type][score_name + "_mean"] = str(round(float(np.mean(score_list)), 2))
        results_dict[model_type][score_name + "_std"] = str(round(float(np.std(score_list)), 2))
            
    save_to_json(dataset, results_dict, exp_num=2, regression=regression)
    return
    
def get_cost(ys, preds):
    costs = np.zeros(preds.shape, dtype=int)
    costs[preds>ys] = 5  #prediction>label -> false positive
    costs[preds<ys] = 1  #prediction<label -> false negative
    return np.mean(costs)

def transform_to_original_scale(predictions_base, predictions_mis, dataset):
    
    return predictions_base, predictions_mis

def mse_loss(predictions, labels):
    return np.mean((predictions-labels)**2)

def avg_diff_per_datapoint(predictions_base, predictions_mis, dataset):
    """ Compute difference (in %) per datapoint and output the average 
        Can be interpreted as "On average, the prediction of the base model 
        is increased/decreased by X percent when compared with the full model """

    predictions_base = undo_transformation(predictions_base, dataset)
    predictions_mis = undo_transformation(predictions_mis, dataset)
    avg_percentage_change = (np.mean(predictions_mis) - np.mean(predictions_base)) / np.absolute(np.mean(predictions_base))
    print("Avg base pred:", np.mean(predictions_base))
    print("Avg mis/resample pred:", np.mean(predictions_mis))
    return avg_percentage_change
    

def get_fnr(confusion_matrix):
    TP = confusion_matrix[1][1]
    FN = confusion_matrix[0][1]
    return FN/(FN+TP)
    
def get_fpr(confusion_matrix):
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[1][0]
    return FP/(FP+TN)

def proba_to_classification(pred):
    # Note: only works for binary classification
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    return pred

def get_dflip_ratios(pred_orig, pred_mis, proba=True):
    # Positive flip := decision is 0 from original model, but 1 from missing model
    # Negative flip := decision is 1 from original model, but 0 from missing model
    if proba:
        pred_orig = proba_to_classification(pred_orig)
        pred_mis = proba_to_classification(pred_mis)
    pos_flip_ratio = np.count_nonzero(pred_orig[pred_mis==1]==0) / len(pred_orig)
    neg_flip_ratio = np.count_nonzero(pred_orig[pred_mis==0]==1) / len(pred_orig)
    return np.array([pos_flip_ratio, neg_flip_ratio])


### Functions to store and save results
def evaluate_and_append_scores(scoring_fns, score_lists, labels, preds):
    """ Calculated the scores and append the results to a dict. """
    for score_name, score_fn in scoring_fns.items():
        score = score_fn(labels, preds)
        if score_name in score_lists:
            score_lists[score_name].append(score)
        else:
            score_lists[score_name] = [score]

def evaluate_and_append_scores_ma(scoring_fns, score_lists, labels_miss, preds_miss, labels_avail, preds_avail):
    """ Calculated the scores and append the results to a dict. """
    for score_name, score_fn in scoring_fns.items():
        for mode in ["miss", "avail", "total"]:
            if mode == "miss":
                labels, preds = labels_miss, preds_miss
            elif mode == "avail":
                labels, preds = labels_avail, preds_avail
            elif mode == "total":
                if labels_miss is not None:
                    labels = np.concatenate([labels_miss, labels_avail])
                else:
                    labels = None
                preds = np.concatenate([preds_miss, preds_avail])
            score = score_fn(labels, preds)
            if score_name + "_" + mode in score_lists:
                score_lists[score_name + "_" + mode].append(score)
            else:
                score_lists[score_name + "_" + mode] = [score]


def aggregate_scores(scoring_fns, score_lists):
    res_dict = {}
    for score_name in score_lists.keys():
        npscore = np.array(score_lists[score_name])
        res_dict[score_name + "_mean"]  = npscore.mean()
        res_dict[score_name + "_std"]  = npscore.std()
        res_dict[score_name + "_all"] = score_lists[score_name] # Dump full list.
    return res_dict
    

def aggregate_and_store_scores(scoring_fns, score_lists, dataset, approach_key, feature_names, experiment_num=2, classifier_class="Logistic"):
    """ Compute mean and stds and store dict """
    res_dict = {"model": classifier_class,
            "voluntary feature": feature_names
    }
    for score_name in scoring_fns.keys():
        npscore = np.array(score_lists[score_name])
        #print(npscore)
        res_dict[score_name + "_mean"]  = npscore.mean()
        res_dict[score_name + "_std"]  = npscore.std()
        res_dict[score_name + "_all"] = score_lists[score_name] # Dump full list.

    save_to_json(dataset, {approach_key: res_dict}, experiment_num)


def smart_update_json(results_dict, file_name):
    """ Update a json file. """
    isExisting = os.path.exists(file_name)
    if isExisting:
        old_res = json.load(open(file_name))
        _rec_update_layer(old_res, results_dict)
        json.dump(old_res, open(file_name,"w"))
    else:
        json.dump(results_dict, open(file_name,"w"))


def _rec_update_layer(old_dict, results_dict):
    for k in results_dict:
        if k in old_dict:
            if type(results_dict[k]) == dict and type(old_dict[k]) == dict:
                print("Recursively updating key", k)
                _rec_update_layer(old_dict[k], results_dict[k])
            else:
                old_dict[k] = results_dict[k]
        else:
            old_dict[k] = results_dict[k] ## Append new results if key does not exist.