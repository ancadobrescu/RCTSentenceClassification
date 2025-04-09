# RCTSentenceClassification

# Project Overview

This repository contains code and data for a reproducibility experiment of two NLP studies on clinical trial sentence classification. 

##  Data Origin and Description

The dataset used is the **CONSORT-TM corpus**, which was made publicly available in both studies referenced in this work. It contains annotated sentences from clinical trial reports, labeled according to CONSORT methodology items.


##  Data Preprocessing

All preprocessing steps—including cleaning, tokenization, and data splitting—are documented in the code (`data_preprocessing/`) and described in the associated article. Intermediate versions of the dataset are also included in the `data/` folder.

##  Experimental Set-up

- The data is split using **5-fold cross-validation** with **StratifiedKFold** to preserve class distributions across folds.
- The exact train/dev/test proportions and the strategy used for class balancing are provided in the article and reflected in the `UMLS_EDA/` folder structure.
- The evaluation metrics used are **precision**, **recall**, and **F1-score**, following **classical definitions** implemented via **scikit-learn**.
- Metric computation is done using `sklearn.metrics` with consistent averaging strategies across experiments.
- The software environment (libraries and versions) is specified in the `requirements.txt` file. The environment can be recreated via:

## Training Process

- Hyperparameters were selected based on values reported in the two original studies, as well as those used in the original GitHub repository for Study 1.
- We used the same settings across all folds for consistency.
- Each model is trained over 30 epochs using 5-fold cross-validation, leading to a total of 5 training and evaluation runs per configuration for 5 seeds.

## Model Evaluation

- All results are reported across folds and seeds.
- Evaluation outputs—including per-fold metrics, predictions, and error analyses are available in the `results/` directory.

##  Folder Structure

### `data/`
- Contains the raw and processed data used across all experiments.
- Files and folders:
  - `50_XML/`: Folder containing 50 XML files that constitute the base corpus used in both studies.
  - `UMLS_EDA/`: Folder containing 5 subfolders (folds), each with:
    - `train.csv`: Training set for the fold.
    - `test.csv`: Test set for the fold.
    - This represents the 5-fold augmented dataset used in Study 2.
  - `Methods_all.csv`: Single fold dataset provided in Study 1, used as the main dataset in our experiments.
  - `Dataset_aug.xlsx`: Augmented dataset created using a threshold of 50 for rare CONSORT items.
  - `Methods_all_wt0.xlsx`: Same as `Methods_all.csv`, but with class 0 removed; used in Experiments 2 and 3.
  - `Dataset_aug_wt0.xlsx`: Augmented dataset excluding class 0, used in Experiments 2 and 3.

### `data_preprocessing/`
- Scripts for cleaning, preprocessing, and formatting data.
- Files:
  - `encoding.py`: Encodes the annotations as a sequence of binary values (1 for the presence of a label, 0 for its absence).
  - `extract_XML.py`: Extracts information from the CONSORT-TM corpus (XML format) to reconstruct the `Methods_all.csv` file originally provided by the two studies.
  - `our_extraction.csv`: Reconstructed `Methods_all.csv` file based on the corpus (XML files).
  - `supp_sentences.csv`: Sentences found in the corpus (our extraction) but not present in the original `Methods_all.csv` file provided by the authors.
  - `transition_labels.csv`: Sentences whose labels differ between the original corpus annotations and the `Methods_all.csv` file provided by the authors.
  - `aug_extraction.xlsx`: Unique sentences extracted from the 5 folds provided in Study 2 for the augmented dataset. Only the training portion is included, as the test set contains the non-augmented sentences without subsection headers.


### `MainExp/`
- Main experiment using full data with the subsection header and full sentence.
- Files:
  - `svm.py`: Runs the SVM model.
  - `biobert_opt.py`: Fine-tunes BioBERT with hyperparameter optimization.
  - `biobert_opt_aug.py`: Same as above with additional data augmentation.

### `Exp1_wtTitle/`
- Experiment without the subsection header information.
- Files:
  - `svm.py`
  - `biobert_opt.py`
  - `biobert_opt_aug.py`

### `Exp2_wt0/`
- Experiment using the subsection header information but without the 0 class.
- Files:
  - `svm.py`
  - `biobert_opt.py`
  - `biobert_opt_aug.py`

### `Exp3_wtTitle_wt0/`
- Experiment without the subsection header information and the 0 class.
- Files:
  - `svm.py`
  - `biobert_opt.py`
  - `biobert_opt_aug.py`

### `results/`
- Contains the results and output metrics of all experiments.
- Subfolders:
  - `MainExp/`
  - `Exp1_wtTitle/`
  - `Exp2_wt0/`
  - `Exp3_wtTitle/`

Each of these contains:
  - `svm/`, `biobert_opt/`, `biobert_opt_aug/`: Model-specific result folders.
    - Inside each model folder:
      - `models/`: Stores all model runs and checkpoints.
      - `fold_X_seed_Y.xlsx`: Excel files representing evaluation results on each of the 5 folds for each random seed used.
      - `all_predictions.xlsx`: Contains all predictions made by the model across folds/seeds.
      - `metrics.xlsx`: Summary of evaluation metrics (e.g., precision, recall, F1-score).
      - `seed_results.xlsx`: Summary of evaluation metrics (e.g., precision, recall, F1-score) for all the seeds.


##  `requirements.txt`
- Contains all Python libraries used in the conda environment for all experiments.
- To recreate the environment, you can run:

```bash
conda create --name <env_name> --file requirements.txt 
```

## Run experiments

### MainExp

To run the main experiment, execute the following commands:

```bash
python MainExp/svm.py
python MainExp/biobert_opt.py
python MainExp/biobert_opt_aug.py
```

### Exp1_wtTitle

```bash
python Exp1_wtTitle/svm.py
python Exp1_wtTitle/biobert_opt.py
python Exp1_wtTitle/biobert_opt_aug.py
```

### Exp2_wt0

```bash
python Exp2_wt0/svm.py
python Exp2_wt0/biobert_opt.py
python Exp2_wt0/biobert_opt_aug.py
```

### Exp3_wtTitle_wt0

```bash
python Exp3_wtTitle_wt0/svm.py
python Exp3_wtTitle_wt0/biobert_opt.py
python Exp3_wtTitle_wt0/biobert_opt_aug.py
```
