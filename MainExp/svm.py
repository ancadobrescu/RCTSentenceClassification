#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import pandas as pd
import pickle
import json
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
import pathlib
import argparse
from scipy.special import expit

# Préparer les stop words
stop_words = set(stopwords.words('english'))
stop_words = list(stop_words)

# Lire et préparer les données
def extract_feature(data_file, save_vocab):
    # Charger les données
    data_df = pd.read_csv(data_file)
    text_data = data_df['top_section'] + " " + data_df['sentence_text']
    text_data.fillna("unknown", inplace=True)
    # Créer les caractéristiques avec TF-IDF
    tfidf = TfidfVectorizer(stop_words=stop_words)
    tfidf_vocab = tfidf.fit(text_data.values.astype('U'))
    text_features = tfidf_vocab.transform(text_data.values.astype('U'))
    # Préparer les étiquettes
    y_labels = data_df['CONSORT_Item']
    y_labels = [
        item.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")
        for item in y_labels
    ]
    labels = ['0', '10', '11a', '11b', '12a', '12b', '3a', '3b', '4a', '4b', '5', '6a', '6b', '7a', '7b', '8a', '8b', '9']
    mlb = MultiLabelBinarizer(classes=labels)
    y_binary = mlb.fit_transform(y_labels)

    if save_vocab:
        vocab_file = "results/MainExp/svm/models/vectorizer.pkl"
        with open(vocab_file, 'wb') as fin:
            pickle.dump(tfidf_vocab, fin)
    return text_features, y_binary, mlb, tfidf_vocab, labels, data_df

def evaluate_model_with_cross_validation(X, y, data_df, mlb_labels, n_splits=5):
    clf = OneVsRestClassifier(LinearSVC(C=10))
    f1_micro_scores = []
    f1_macro_scores = []
    precision_micro_scores = []
    recall_micro_scores = []
    precision_macro_scores = []
    recall_macro_scores = []
    accuracy_scores = []
    y_true_all = []
    y_pred_all = []
    f1_macro_per_fold = [] 
    error_counts = Counter() 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Initialisation de excel_results et mismatch_results
    excel_results = []
    mismatch_results = []
    all_indices = []  

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y.argmax(axis=1))):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        clf.fit(X_train_fold, y_train_fold)
        # Ajout du calcul des scores bruts avant prédiction
        y_scores = clf.decision_function(X_val_fold)
        y_probabilities = expit(y_scores)
        y_pred = clf.predict(X_val_fold)
        # Ajouter les vrais labels et les prédictions pour toutes les itérations
        y_true_all.append(y_val_fold)
        y_pred_all.append(y_pred)
        all_indices.extend(val_idx) 
        # Calcul des métriques
        f1_micro = f1_score(y_val_fold, y_pred, average='micro')
        f1_macro = f1_score(y_val_fold, y_pred, average='macro')
        precision_micro = precision_score(y_val_fold, y_pred, average='micro')
        recall_micro = recall_score(y_val_fold, y_pred, average='micro')
        precision_macro = precision_score(y_val_fold, y_pred, average='macro')
        recall_macro = recall_score(y_val_fold, y_pred, average='macro')
        accuracy = accuracy_score(y_val_fold, y_pred)
        # Stocker les scores
        f1_micro_scores.append(f1_micro)
        f1_macro_scores.append(f1_macro)
        precision_micro_scores.append(precision_micro)
        recall_micro_scores.append(recall_micro)
        precision_macro_scores.append(precision_macro)
        recall_macro_scores.append(recall_macro)
        accuracy_scores.append(accuracy)
        # Enregistrer l'évolution de F1-macro
        f1_macro_per_fold.append(f1_macro)

        # Comparaison des prédictions et des valeurs réelles
        for true_labels, pred_labels, row in zip(y_val_fold, y_pred, data_df.iloc[val_idx].itertuples()):
            true_labels_str = [mlb_labels.classes_[i].strip() for i in range(len(true_labels)) if true_labels[i] == 1]
            pred_labels_str = [mlb_labels.classes_[i].strip() for i in range(len(pred_labels)) if pred_labels[i] == 1]
            prediction_output = ', '.join(pred_labels_str) if pred_labels_str else 'no_prediction'
            match = "yes" if set(true_labels_str) == set(pred_labels_str) else "no"
            match_type = "good_prediction" if match == "yes" else "mismatch"
            excel_results.append({
                "original_value": ', '.join(true_labels_str),
                "prediction": prediction_output,
                "match": match,
                "PMCID": row.PMCID,
                "sentence_text": row.sentence_text,
                "top_section": row.top_section
            })
            if match == "no":
                mismatch_results.append({
                    "original_value": ', '.join(true_labels_str),
                    "prediction": prediction_output,
                    "type_mismatch": match_type,
                    "PMCID": row.PMCID,
                    "sentence_text": row.sentence_text,
                    "top_section": row.top_section
                })

    # Alignement correct de `data_df`
    data_df_aligned = data_df.iloc[all_indices].reset_index(drop=True)
    # Sauvegarde des mismatches dans un fichier Excel
    mismatch_df = pd.DataFrame(mismatch_results)
    freq_df = mismatch_df.groupby(['original_value', 'prediction']).size().reset_index(name='frequency')
    freq_df.to_excel("results/MainExp/svm/mismatches_frequency.xlsx", index=False)
    # Sauvegarde des prédictions dans un fichier Excel
    val_df = pd.DataFrame(excel_results)
    val_df.to_excel("results/MainExp/svm/predictions_original_values.xlsx", index=False)
    y_true_all = np.vstack(y_true_all)  
    y_pred_all = np.vstack(y_pred_all)
    class_report = classification_report(y_true_all, y_pred_all, target_names=mlb_labels.classes_)
    print("\nReport classification for each class :")
    print(class_report)

    return f1_macro_per_fold, error_counts, y_true_all, y_pred_all

if __name__ == "__main__":
    train_data_file = "data/Methods_all.csv"
    parser = argparse.ArgumentParser(description='Training. Define the parameters.')
    parser.add_argument("--data_file", default=train_data_file, required=False)
    parser.add_argument("--override_vocab", default=True, required=False)
    args = parser.parse_args()
    # Charger et préparer les données
    X, y, mlb_labels, tfidf_vocab, labels, data_df = extract_feature(args.data_file, args.override_vocab)
    print("5-fold cross-validation in progress...")
    f1_macro_per_fold, error_counts, y_true_all, y_pred_all = evaluate_model_with_cross_validation(X, y, data_df, mlb_labels)
    final_clf = OneVsRestClassifier(LinearSVC(C=10)).fit(X, y)
    model_filename = 'results/MainExp/svm/models/final_model.sav'
    pickle.dump({'clf': final_clf, 'binarizer': mlb_labels, 'vectorizer': tfidf_vocab}, open(model_filename, 'wb'))
    # Récupérer les scores bruts de classification
    y_scores = final_clf.decision_function(X) 
    # Transformer en pseudo-probabilités avec une sigmoïde
    y_probabilities = expit(y_scores)
    excel_results = []

    for i, row in enumerate(data_df.itertuples()):
        true_labels = [labels[j] for j in range(len(labels)) if y_true_all[i, j] == 1]
        pred_labels = [labels[j] for j in range(len(labels)) if y_pred_all[i, j] == 1]
        # Vérification des valeurs réelles et des prédictions
        if not true_labels:
            true_labels_str = "No"  
        else:
            true_labels_str = ', '.join(true_labels)
        if not pred_labels:
            pred_labels_str = "No" 
        else:
            pred_labels_str = ', '.join(pred_labels)
        # Créer le dictionnaire pour les probabilités de chaque label
        prob_dict = {f"proba_{labels[j]}": y_probabilities[i, j] for j in range(len(labels))}
        # Ajouter les résultats 
        excel_results.append({
            "PMCID": row.PMCID,
            "sentence_text": row.sentence_text,
            "top_section": row.top_section,
            "original_value": true_labels_str,
            "prediction": pred_labels_str,
            "match": "yes" if set(true_labels) == set(pred_labels) else "no",
            **prob_dict
        })
    val_df = pd.DataFrame(excel_results)
    val_df.to_excel("results/MainExp/svm/predictions_with_probabilities.xlsx", index=False)

