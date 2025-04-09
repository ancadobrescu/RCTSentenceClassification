import csv
from collections import Counter, defaultdict
import re
import pandas as pd
from ast import literal_eval
import nltk
import openpyxl

input_csv = "data/Methods_all_wt0.xlsx"
# Charger le fichier CSV (assurez-vous que le chemin vers votre fichier est correct)
df = pd.read_excel(input_csv)
print(df.columns)

# Extraire les catégories uniques après nettoyage, en maintenant l'ordre et en incluant la classe "0"
categories = sorted(set([item for sublist in df['CONSORT_Item']
                         .str.strip("[]").str.replace("'", "").str.split(', ')
                         for item in sublist]))

print(f"Nombre de catégories : {len(categories)}")  
print(f"Catégories : {categories}")  # Vérifier l'ordre

# Fonction pour générer des labels binaires en respectant l'ordre des catégories
def generate_labels(consort_item_value, categories):
    consort_items = consort_item_value.strip("[]").replace("'", "").split(', ')
    labels = [1 if category in consort_items else 0 for category in categories]
    return labels

# Appliquer la fonction pour générer des labels binaires
df['labels'] = df['CONSORT_Item'].apply(lambda x: generate_labels(x, categories))

# Vérifier la longueur des labels pour chaque ligne
print(df['labels'].apply(len).unique())  

# Prétraitement : transformer le texte en minuscules
df['text_orig'] = df['text_orig'].astype(str)  # S'assurer que 'sentence_text' est bien une chaîne
df['text_orig'] = df['text_orig'].str.lower()  # Convertir en minuscules

# Remplacer les retours à la ligne, tabulations, astérisques et autres caractères indésirables par des espaces
df['text_orig'] = df['text_orig'].replace({r'\n': ' ', r'\r': ' ', r'\t': ' ', r'\*': ' '}, regex=True)

# Nettoyer les espaces en début et fin de chaîne
df['text_orig'] = df['text_orig'].str.strip()

# Remplacer les virgules et guillemets qui peuvent être mal interprétés dans le CSV
df['text_orig'] = df['text_orig'].replace({r',': ' ', r'"': ' '}, regex=True)

# Prétraitement : transformer le texte en minuscules
"""df['sentence_text'] = df['sentence_text'].astype(str)  # S'assurer que 'sentence_text' est bien une chaîne
df['sentence_text'] = df['sentence_text'].str.lower()  # Convertir en minuscules

# Remplacer les retours à la ligne, tabulations, astérisques et autres caractères indésirables par des espaces
# df['sentence_text'] = df['sentence_text'].replace({r'\n': ' ', r'\r': ' ', r'\t': ' ', r'\*': ' '}, regex=True)

# Nettoyer les espaces en début et fin de chaîne
df['sentence_text'] = df['sentence_text'].str.strip()

# Remplacer les virgules et guillemets qui peuvent être mal interprétés dans le CSV
df['sentence_text'] = df['sentence_text'].replace({r',': ' ', r'"': ' '}, regex=True)

# Prétraitement : transformer le texte en minuscules
df['top_section'] = df['top_section'].astype(str)  # S'assurer que 'sentence_text' est bien une chaîne

# Remplacer les retours à la ligne, tabulations, astérisques et autres caractères indésirables par des espaces
# df['top_section'] = df['top_section'].replace({r'\n': ' ', r'\r': ' ', r'\t': ' ', r'\*': ' '}, regex=True)

# Nettoyer les espaces en début et fin de chaîne
df['top_section'] = df['top_section'].str.strip()

# Remplacer les virgules et guillemets qui peuvent être mal interprétés dans le CSV
df['top_section'] = df['top_section'].replace({r',': ' ', r'"': ' '}, regex=True)"""

# Vérification de l'alignement des colonnes avant et après transformation
print("Colonnes avant transformation:", df.columns)

# Afficher un échantillon du DataFrame transformé
print(df[['PMCID', 'sentence_id', 'text_orig', 'CONSORT_Item', 'labels', 'augment', 'methods']].head())
#print(df[['PMCID', 'sentence_id', 'sentence_text', 'top_section', 'CONSORT_Item', 'labels']].head())


# Vérification de la forme du DataFrame après transformation
print("Forme du DataFrame après transformation:", df.shape)

# Enregistrer le DataFrame propre dans un fichier CSV
output_csv_sentences = "data/Methods_all_wt0.xlsx"

# Exporter en CSV, en utilisant des guillemets pour gérer les colonnes contenant des virgules et des espaces
df.to_excel(output_csv_sentences, index=False)
