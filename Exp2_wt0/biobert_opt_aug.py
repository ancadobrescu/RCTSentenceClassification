import pandas as pd
import numpy as np
import torch
from simpletransformers.classification import MultiLabelClassificationModel
from ast import literal_eval
import itertools
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import openpyxl
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer
import random

# Arguments du script
parser = argparse.ArgumentParser(description='Multi Label Classification')
parser.add_argument('--data_path', help='path to .xlsx consisting of sentences', default="data/Dataset_aug_wt0.xlsx")
parser.add_argument('--model_path', help='path to directory consisting of BERT model', default="dmis-lab/biobert-base-cased-v1.2")
parser.add_argument('--model_save_path', help='path to directory consisting of BERT model', default="results/Exp2_wt0/biobert_opt_aug//models")
parser.add_argument('--text_column', help='column name containing sentences', default="text")

args = vars(parser.parse_args())

data_path = args["data_path"]
model_path = args["model_path"]
text_col = args["text_column"]
model_save_path = args["model_save_path"]

if model_save_path is None:
    print("Please enter path to save the model in training mode")
    exit(0)

# Charger les données
train_df_linh = pd.read_excel(data_path)
train_df_linh["CONSORT_Item"] = train_df_linh["CONSORT_Item"].apply(lambda x: literal_eval(x))
train_df_linh["labels"] = train_df_linh["labels"].apply(lambda x: literal_eval(x))

# One-hot encoding de la colonne top_section
one_hot_top_section = pd.get_dummies(train_df_linh["top_section"], prefix="top_section")

# Créer une liste binaire de top_section
train_df_linh["top_section_binary"] = one_hot_top_section.apply(lambda row: row.tolist(), axis=1)

# Fusionner les colonnes de texte
train_df_linh["text"] = train_df_linh["text_orig"]

# Ajouter les colonnes one-hot encodées et la colonne binaire comme traits distincts
train_df_linh = pd.concat([train_df_linh, one_hot_top_section], axis=1)

# Préparer les labels
consort_items = sorted(set(itertools.chain(*train_df_linh["CONSORT_Item"])))
id2index = {i: item for i, item in enumerate(consort_items)}
n_labels = len(id2index)

mlb = MultiLabelBinarizer(classes=consort_items)
train_df_linh["labels"] = list(mlb.fit_transform(train_df_linh["CONSORT_Item"]))

# Convertir les labels en une forme compatible pour StratifiedKFold
train_df_linh["label_tuple"] = train_df_linh["labels"].apply(lambda x: tuple(x))
y = [sum([2**i if label == 1 else 0 for i, label in enumerate(labels)]) for labels in train_df_linh["labels"]]

# Vérification si CUDA est disponible
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# Configuration du modèle
config = {
    'reprocess_input_data': True,
    'fp16': False,
    'evaluate_during_training': False,
    'output_dir': model_save_path,
    'train_batch_size': 4,
    'gradient_accumulation_steps': 1,
    'learning_rate': 3e-5,
    'num_train_epochs': 30,
    'overwrite_output_dir': True,
    'do_lower_case': True,
    'max_seq_length': 512,
    'threshold': 0.4,
    'dropout_rate': 0.1,
    'use_cuda': torch.cuda.is_available(),
    'logging_steps': 500, 
    'use_progress_bar': True,
    'n_gpu': 3
}

# Liste de seeds plus variés
seeds = [42, 1234, 5678, 9012, 9999]

fold_no = 1

# Pour stocker les résultats globaux
precision_macro = []
recall_macro = []
f1_macro = []
precision_micro = []
recall_micro = []
f1_micro = []
roc_auc_scores = []
# Stocker toutes les prédictions
all_predictions = []

# Pour stocker les résultats détaillés par seed et fold
all_seed_results = []

# Pour stocker les résultats par classe (pour chaque pli)
per_class_metrics = {item: {'precision': [], 'recall': [], 'f1-score': [], 'support': [], 'roc_auc': []} for item in consort_items}

# Séparer les phrases avec augment == 1
always_train_df = train_df_linh[train_df_linh["augment"] == 1].copy()

# Séparer les phrases avec augment == 0
other_df = train_df_linh[train_df_linh["augment"] == 0].copy()

# Assurez-vous que PMCID et sentence_id sont des types hachables (chaînes ou tuples)
always_train_df["PMCID"] = always_train_df["PMCID"].apply(str)
always_train_df["sentence_id"] = always_train_df["sentence_id"].apply(str)

other_df["PMCID"] = other_df["PMCID"].apply(str)
other_df["sentence_id"] = other_df["sentence_id"].apply(str)

# Convertir les colonnes contenant des listes ou des tableaux en tuples pour être hachables
always_train_df = always_train_df.applymap(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
other_df = other_df.applymap(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

# Créer un ensemble des clés uniques des phrases augmentées (augment == 1)
train_keys = set(zip(always_train_df["PMCID"], always_train_df["sentence_id"]))

# Filtrer les phrases augmentées avec augment == 0 qui ont les mêmes clés que celles avec augment == 1
matching_augment_0 = other_df[other_df.apply(lambda row: (row["PMCID"], row["sentence_id"]) in train_keys, axis=1)]

# Ajouter ces correspondances dans train_df
always_train_df = pd.concat([always_train_df, matching_augment_0]).drop_duplicates()

# Supprimer ces correspondances de other_df (pour éviter qu'elles y soient incluses)
other_df = other_df[~other_df.index.isin(matching_augment_0.index)]

# Afficher la taille des DataFrames finaux
print(f"Taille du train final : {always_train_df.shape}")
print(f"Taille du reste des données (other_df) : {other_df.shape}")

# Recalculer les labels pour la stratification
y_other = [
    sum([2**i if label == 1 else 0 for i, label in enumerate(labels)])
    for labels in other_df["labels"]
]

# Effectuer la validation croisée sur other_df uniquement
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

try:
    # Vérifier si '6b' est dans la liste de CONSORT_Item pour les lignes où augment == 0
    forced_example_index = other_df[(other_df['augment'] == 0) & (other_df['CONSORT_Item'].apply(lambda x: '6b' in x))].index[0]
    print(f"L'index du premier exemple où augment == 0 et CONSORT_Item == '6b' est : {forced_example_index}")
except IndexError:
    print("Aucun exemple ne correspond aux critères spécifiés.")
    forced_example_index = None 

# Calcul des métriques pour chaque seed
for seed in seeds:
    print(f"Running with seed {seed}...")
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_results=[]
    fold_no = 1
    # Calcul des métriques pour chaque pli
    for train_index, val_index in skf.split(np.zeros(len(y_other)), y_other):
        # Vérification que l'exemple forcé est bien dans `other_df`
        if forced_example_index not in other_df.index:
            print(f"Attention : l'index {forced_example_index} n'est pas dans `other_df`.")
        else:
            print(f"L'index {forced_example_index} est bien dans `other_df`.")
        # Séparer les données avant l'ajout forcé
        train_df = other_df.iloc[train_index]
        val_df = other_df.iloc[val_index]
        # Ajouter always_train_df à train_df
        train_df = pd.concat([train_df, always_train_df]).drop_duplicates()
        # Vérification avant retrait : afficher l'indice forcé dans train_df
        print(f"\nAvant retrait de l'exemple forcé, train_df contient l'exemple avec l'index {forced_example_index} :")
        print(train_df[train_df.index == forced_example_index])
        # Retirer l'exemple forcé de train_df
        train_df = train_df[train_df.index != forced_example_index] 
        print(f"\ntrain_df après retrait de l'exemple forcé (index {forced_example_index}) :")
        print(train_df[train_df.index == forced_example_index]) 
        # Ajouter l'exemple forcé à val_df
        val_df = pd.concat([val_df, other_df.loc[[forced_example_index]]]) 
        # Vérification spécifique de l'exemple forcé dans val_df après l'ajout forcé
        print(f"\nContenu de val_df après ajout forcé :")
        if forced_example_index in val_df.index:
            print(f"L'exemple avec l'index {forced_example_index} est bien dans val_df après ajout forcé.")
        else:
            print(f"L'exemple avec l'index {forced_example_index} n'est pas dans val_df après ajout forcé.")
        # Vérification des indices de validation dans other_df
        print(f"\nIndices de validation absents de 'other_df' : {set(val_index) - set(other_df.index)}")
        # Re-séparer les données après l'ajout forcé
        train_df = pd.concat([train_df, other_df.iloc[train_index]]).drop_duplicates()  
        val_df = pd.concat([val_df, other_df.iloc[val_index]]).drop_duplicates()
        # Vérification spécifique de l'exemple forcé dans val_df après re-séparation
        print(f"\nContenu de val_df après re-séparation :")
        if forced_example_index in val_df.index:
            print(f"L'exemple avec l'index {forced_example_index} est toujours dans val_df après re-séparation.")
        else:
            print(f"L'exemple avec l'index {forced_example_index} n'est pas dans val_df après re-séparation.")
        # Vérification avant retrait : afficher l'indice forcé dans train_df
        print(f"\nAvant retrait de l'exemple forcé, train_df contient l'exemple avec l'index {forced_example_index} :")
        print(train_df[train_df.index == forced_example_index])
        # Retirer l'exemple forcé de train_df
        train_df = train_df[train_df.index != forced_example_index] 
        print(f"\ntrain_df après retrait de l'exemple forcé (index {forced_example_index}) :")
        print(train_df[train_df.index == forced_example_index])  
        # Afficher les indices de validation après la re-séparation
        print(f"\nIndices de validation après re-séparation : {val_df.index.tolist()}")
        # Afficher les indices de train après la re-séparation
        print(f"\nIndices de train après re-séparation : {train_df.index.tolist()}")
        print(f"Starting training for fold {fold_no}")
        # Initialiser le modèle pour chaque pli
        model = MultiLabelClassificationModel('bert', model_path, num_labels=n_labels, args=config)
        # Initialiser l'optimiseur Adam
        optimizer = torch.optim.Adam(model.model.parameters(), lr=config['learning_rate'])
        # Vérifier que les données avec augment == 1 sont bien dans train
        print(f"\n[Fold {fold_no}] Vérification de la répartition:")
        print(f"Train size: {len(train_df)}, Test size: {len(val_df)}")
        print(f"Train indices: {train_index[:5]}...")
        print(f"Validation indices: {val_index[:5]}...")
        # Vérification de la distribution des classes dans chaque pli
        train_class_distribution = np.sum(np.array(train_df["labels"].tolist()), axis=0)
        val_class_distribution = np.sum(np.array(val_df["labels"].tolist()), axis=0)
        print(f"Train class distribution for fold {fold_no}:")
        for class_name, count in zip(consort_items, train_class_distribution):
            print(f"Class '{class_name}': {count} occurrences")
        print(f"Validation class distribution for fold {fold_no}:")
        for class_name, count in zip(consort_items, val_class_distribution):
            print(f"Class '{class_name}': {count} occurrences")

        # Sauvegarder les fichiers Excel pour la répartition des folds
        with pd.ExcelWriter(f"results/Exp2_wt0/biobert_opt_aug/train_fold_{fold_no}_seed_{seed}.xlsx") as writer:
            train_df.to_excel(writer, index=False, sheet_name="Train Data")

        with pd.ExcelWriter(f"results/Exp2_wt0/biobert_opt_aug/test_fold_{fold_no}_seed_{seed}.xlsx") as writer:
            val_df.to_excel(writer, index=False, sheet_name="Test Data")

        # Calculer le nombre total de pas d'entraînement
        total_steps = len(train_df) * config['num_train_epochs']
        # Ajouter un scheduler pour ajuster le taux d'apprentissage pendant l'entraînement
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        # Entraînement avec l'optimiseur et scheduler
        model.train_model(train_df, optimizer=optimizer, scheduler=scheduler)
        # Prédictions sur l'ensemble de validation
        predictions, raw_outputs = model.predict(val_df['text'].tolist())
        # Conversion des probabilités
        probabilities = predictions
        # Construire le DataFrame des résultats
        results_df = val_df.copy()
        results_df['fold'] = fold_no
        # S'assurer que les prédictions et les labels sont alignés
        true_labels = [
            [consort_items[i] for i, value in enumerate(label) if value == 1]
            for label in val_df["labels"].tolist()
        ]
        # S'assurer que les prédictions sont obtenues correctement pour chaque exemple
        predicted_classes = [
            [consort_items[i] for i, value in enumerate(prediction) if value == 1]
            for prediction in predictions
        ]
        # Vérifier les tailles des objets avant d'affecter à results_df
        print(f"Size of true_labels: {len(true_labels)}")
        print(f"Size of predicted_classes: {len(predicted_classes)}")
        print(f"Size of results_df: {len(results_df)}")
        # Vérification des tailles des résultats après la prédiction
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of true labels: {len(val_df)}")
        # Comparer true_labels et predicted_classes
        if len(true_labels) == len(results_df):
            match = [set(true) == set(pred) for true, pred in zip(true_labels, predicted_classes)]
        else:
            print("Mismatched lengths between true labels and results_df!")
            match = []
        # Maintenant, on peut ajouter la colonne match au DataFrame
        results_df['match'] = match
        # Vérifier que la longueur de 'match' correspond à la longueur de 'results_df'
        print(f"Length of match: {len(match)}")
        print(f"Length of results_df: {len(results_df)}")
        # Ajouter les prédictions et labels à results_df
        results_df['predicted_labels'] = predicted_classes
        results_df['true_labels'] = true_labels
        # S'assurer que les colonnes de probabilité sont dans le bon ordre
        prob_df = pd.DataFrame(probabilities, columns=consort_items)
        prob_df = prob_df[consort_items] 
        # Vérification de la taille des DataFrames
        print(f"Size of results_df: {results_df.shape}")
        print(f"Size of prob_df: {prob_df.shape}")
        # Vérification de l'alignement des classes
        assert list(prob_df.columns) == list(consort_items), "Il y a une discordance dans l'ordre des classes!"
        # Avant la concaténation des DataFrames, réinitialiser les index
        results_df = results_df.reset_index(drop=True)
        prob_df = prob_df.reset_index(drop=True)
        # Concaténer les probabilités au DataFrame des résultats
        results_df = pd.concat([results_df, prob_df], axis=1)
        # Stocker les prédictions de ce pli
        all_predictions.append(results_df)
        print(f"Fold {fold_no} stockage prédictions terminé.\n")

        # Boucle pour imprimer le taux d'apprentissage après chaque mise à jour
        for epoch in range(config['num_train_epochs']):
            # Imprimer le taux d'apprentissage actuel avec toutes les décimales
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{config['num_train_epochs']}: Current learning rate: {current_lr:.20f}")
        # Entraînement
        print(f"Finished training for fold {fold_no}. Evaluating now...")
        
        # Évaluer le modèle sur le pli de validation
        result, model_outputs, wrong_predictions = model.eval_model(val_df)
        # Collecte des métriques
        y_true = np.array(val_df["labels"].tolist())
        y_pred = (model_outputs > config['threshold']).astype(int)
        match = np.all(y_true == y_pred, axis=1)
        match = np.where(match, 'oui', 'non')

        # Fonction pour obtenir les valeurs de base à partir de l'encodage binaire
        def get_labels_from_encoding(labels, label_list):
            return [label_list[i] for i in range(len(labels)) if labels[i] == 1]
        
        # Calcul des métriques macro et micro
        precision_macro_fold = precision_score(y_true, y_pred, average='macro')
        recall_macro_fold = recall_score(y_true, y_pred, average='macro')
        f1_macro_fold = f1_score(y_true, y_pred, average='macro')
        precision_micro_fold = precision_score(y_true, y_pred, average='micro')
        recall_micro_fold = recall_score(y_true, y_pred, average='micro')
        f1_micro_fold = f1_score(y_true, y_pred, average='micro')

        # Calcul du ROC AUC (en vérifiant s'il y a plusieurs classes)
        if len(np.unique(y_true)) > 1:  
            roc_auc_fold = roc_auc_score(y_true, model_outputs, average='macro', multi_class='ovr')
        else:
            roc_auc_fold = None  

        # Ajouter les résultats pour chaque pli
        precision_macro.append(precision_macro_fold)
        recall_macro.append(recall_macro_fold)
        f1_macro.append(f1_macro_fold)
        precision_micro.append(precision_micro_fold)
        recall_micro.append(recall_micro_fold)
        f1_micro.append(f1_micro_fold)
        roc_auc_scores.append(roc_auc_fold)

        # Calcul des métriques par classe
        for i, class_name in enumerate(consort_items):
            precision_class = precision_score(y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
            recall_class = recall_score(y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
            f1_class = f1_score(y_true[:, i], y_pred[:, i], average='binary', zero_division=0)
            support_class = np.sum(y_true[:, i])

            # Calculer le ROC AUC uniquement si plusieurs classes sont présentes dans y_true et y_pred
            if len(np.unique(y_true[:, i])) > 1 and len(np.unique(y_pred[:, i])) > 1:
                roc_auc_class = roc_auc_score(y_true[:, i], model_outputs[:, i])
            else:
                roc_auc_class = None  

            per_class_metrics[class_name]['precision'].append(precision_class)
            per_class_metrics[class_name]['recall'].append(recall_class)
            per_class_metrics[class_name]['f1-score'].append(f1_class)
            per_class_metrics[class_name]['support'].append(support_class)
            per_class_metrics[class_name]['roc_auc'].append(roc_auc_class)

        # Ajouter les résultats pour chaque pli
        seed_results.append({
            'seed': seed,
            'fold': fold_no,
            'precision_macro': precision_macro_fold,
            'recall_macro': recall_macro_fold,
            'f1_macro': f1_macro_fold,
            'precision_micro': precision_micro_fold,
            'recall_micro': recall_micro_fold,
            'f1_micro': f1_micro_fold,
            'roc_auc': roc_auc_fold
        })
        fold_no += 1
    all_seed_results.extend(seed_results)

# Convertir les résultats des seeds en DataFrame
seed_results_df = pd.DataFrame(all_seed_results)

# Sauvegarder les résultats des seeds dans un fichier Excel
seed_results_df.to_excel("results/Exp2_wt0/biobert_opt_aug/seed_results.xlsx", index=False)

# Fusionner toutes les prédictions en un seul fichier Excel
final_predictions = pd.concat(all_predictions, ignore_index=True)
final_predictions.to_excel("results/Exp2_wt0/biobert_opt_aug/all_predictions.xlsx", index=False)

# Calcul des moyennes des métriques sur les 5 plis
avg_precision_macro = np.mean(precision_macro)
avg_recall_macro = np.mean(recall_macro)
avg_f1_macro = np.mean(f1_macro)
avg_precision_micro = np.mean(precision_micro)
avg_recall_micro = np.mean(recall_micro)
avg_f1_micro = np.mean(f1_micro)
avg_roc_auc = np.mean([score for score in roc_auc_scores if score is not None])  # Ignorer les None

# Création du DataFrame pour les métriques
columns = ['precision', 'recall', 'f1-score', 'support', 'roc_auc']
metrics_data = []

# Ajouter les résultats par classe dans le DataFrame
for class_name in consort_items:
    metrics_data.append([
        np.mean(per_class_metrics[class_name]['precision']),
        np.mean(per_class_metrics[class_name]['recall']),
        np.mean(per_class_metrics[class_name]['f1-score']),
        np.sum(per_class_metrics[class_name]['support']),
        np.mean([roc_auc for roc_auc in per_class_metrics[class_name]['roc_auc'] if roc_auc is not None])
    ])

# Ajouter les moyennes micro et macro
metrics_data.append([
    avg_precision_micro, avg_recall_micro, avg_f1_micro, np.sum([sum(per_class_metrics[class_name]['support']) for class_name in consort_items]), avg_roc_auc
])

metrics_data.append([
    avg_precision_macro, avg_recall_macro, avg_f1_macro, np.sum([sum(per_class_metrics[class_name]['support']) for class_name in consort_items]), avg_roc_auc
])

# Convertir les résultats en DataFrame
metrics_df = pd.DataFrame(metrics_data, columns=columns, index=consort_items + ['micro avg', 'macro avg'])

# Sauvegarder les résultats dans un fichier Excel
metrics_df.to_excel("results/Exp2_wt0/biobert_opt_aug/cross_validation_metrics.xlsx")

# Sauvegarder les résultats finaux dans un fichier Excel
final_metrics = {
    'Metric': ['Precision Macro', 'Recall Macro', 'F1 Macro', 'Precision Micro', 'Recall Micro', 'F1 Micro', 'ROC AUC'],
    'Mean': [avg_precision_macro, avg_recall_macro, avg_f1_macro, avg_precision_micro, avg_recall_micro, avg_f1_micro, avg_roc_auc]
}

final_metrics_df = pd.DataFrame(final_metrics)
final_metrics_df.to_excel("results/Exp2_wt0/biobert_opt_aug/final_metrics.xlsx", index=False)

# Boxplot pour la variabilité entre les seeds
plt.figure(figsize=(8, 6))
plt.boxplot(np.array(f1_macro).reshape(len(seeds), 5), labels=[f'Seed {i + 1}' for i in range(5)])
plt.title('F1_macro Score Variability Across Seeds')
plt.xlabel('Seed')
plt.ylabel('F1 Score')
plt.grid(True)
plt.savefig("results/Exp2_wt0/biobert_opt_aug/f1_variability_seeds.png")
plt.show()