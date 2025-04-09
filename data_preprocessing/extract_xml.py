import os
import xml.etree.ElementTree as ET
import csv

# Fonction pour extraire les informations d'un fichier XML
def extract_info_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extraire le PMCID depuis l'attribut id de la balise <document>
    pmcid = root.attrib.get("id", "")
    sentences = []

    # Extraire le texte complet du document pour l'utiliser plus tard
    full_text = ""
    for text_element in root.findall(".//text"):
        full_text += text_element.text.strip() if text_element.text else ""

    # Trouver toutes les sections avec leurs titres
    section_titles = []
    for section in root.findall(".//section"):
        title = section.attrib.get("title", "")
        start, end = map(int, section.attrib.get("textSpan", "0-0").split("-"))
        section_titles.append({
            "title": title,
            "start": start,
            "end": end,
            "subsections": []  # Pour contenir les sous-sections
        })

    # Ajouter les sous-sections dans les sections correspondantes
    for section in section_titles:
        for sub_section in root.findall(f".//section[@textSpan='{section['start']}-{section['end']}']/section"):
            sub_title = sub_section.attrib.get("title", "")
            sub_start, sub_end = map(int, sub_section.attrib.get("textSpan", "0-0").split("-"))
            section["subsections"].append({
                "title": sub_title,
                "start": sub_start,
                "end": sub_end
            })

    # Fonction pour compter le nombre de tokens dans une phrase
    def count_tokens(sentence_text):
        return len(sentence_text.split())

    # Fonction pour compter le nombre de valeurs non vides dans CONSORT_Item
    def count_non_empty_CONSORT_items(consort_item):
        return len([item for item in consort_item if item.strip()])

    # Parcourir les balises <sentence> pour récupérer les informations des phrases
    for sentence in root.findall(".//sentence"):
        sentence_id = sentence.attrib.get("id", "")
        selection = sentence.attrib.get("selection", "").split(",")  # Séparer par des virgules si plusieurs valeurs
        char_offset = sentence.attrib.get("charOffset", "")

        # Extraire la portion de texte de la phrase en utilisant charOffset
        start, end = map(int, char_offset.split('-'))
        sentence_text = full_text[start:end].strip()

        # Trouver le titre de la section pour cette phrase
        section_title = "Unknown Section"  # Valeur par défaut
        found_section = False

        # Vérifier les sous-sections d'abord
        for section in section_titles:
            for subsection in section["subsections"]:
                if subsection["start"] <= start <= subsection["end"]:
                    section_title = subsection["title"]
                    found_section = True
                    break
            if found_section:
                break
        
        # Si la phrase n'est pas trouvée dans les sous-sections, vérifier la section principale
        if not found_section:
            for section in section_titles:
                if section["start"] <= start <= section["end"]:
                    section_title = section["title"]
                    break

        # Nettoyer les données pour éviter les espaces superflus
        selection = [item.strip() for item in selection if item.strip()]  # Nettoyer la liste des CONSORT_Item
        sentence_text = sentence_text.strip()  # Nettoyer le texte de la phrase
        section_title = section_title.strip()  # Nettoyer le titre de la section

        # Ajouter les informations de chaque phrase dans la liste
        sentences.append({
            "PMCID": pmcid,
            "sentence_id": sentence_id,
            "top_section": section_title,
            "CONSORT_Item": selection,
            "sentence_text": sentence_text,
        })

    return sentences

# Fonction pour parcourir tous les fichiers XML dans un dossier
def process_xml_files_in_directory(directory_path):
    all_sentences = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if file_path.endswith(".xml") and os.path.isfile(file_path):
            sentences = extract_info_from_xml(file_path)
            all_sentences.extend(sentences)
    
    return all_sentences

# Fonction pour afficher les résultats sous forme tabulaire (pour vérification)
def display_sentences(sentences):
    print("PMCID | sentence_id | top_section | CONSORT_Item | sentence_text")
    for sentence in sentences[:5]:  # Affiche les 5 premières phrases pour vérifier
        print(f"{sentence['PMCID']} | {sentence['sentence_id']} | {sentence['top_section']} | {sentence['CONSORT_Item']} | {sentence['sentence_text']}")

# Fonction pour écrire les résultats dans un fichier CSV
def write_sentences_to_csv(sentences, output_csv):
    # Définir l'en-tête du CSV
    fieldnames = ["PMCID", "sentence_id", "top_section", "CONSORT_Item", "sentence_text"]

    # Ouvrir le fichier CSV en mode écriture
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Écrire l'en-tête dans le fichier CSV
        writer.writeheader()

        # Vérifier la consistance des données avant de les écrire
        for sentence in sentences:
            # Afficher un échantillon pour vérifier la consistance
            if len(sentence) != len(fieldnames):
                print(f"Problème avec la ligne : {sentence}")  # Affiche la ligne qui ne correspond pas
            writer.writerow(sentence)

# Spécifiez le dossier contenant vos fichiers XML
directory_path = "data/50_XML" 

# Processer les fichiers XML et extraire les informations
sentences = process_xml_files_in_directory(directory_path)

# Affichage du nombre de phrases extraites
print(f"Nombre total de phrases extraites : {len(sentences)}")

# Afficher un échantillon de quelques phrases pour vérifier
if len(sentences) > 0:
    print(f"Exemple de phrases extraites : {sentences[:5]}")  # Affiche les 5 premières phrases extraites

# Spécifiez le chemin du fichier CSV de sortie pour les phrases
output_csv_sentences = "data_preprocessing/our_extraction.csv"  

# Écrire les résultats des phrases dans le fichier CSV
write_sentences_to_csv(sentences, output_csv_sentences)

