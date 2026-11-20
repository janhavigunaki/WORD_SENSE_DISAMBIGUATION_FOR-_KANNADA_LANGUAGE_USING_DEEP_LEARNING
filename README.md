# WORD_SENSE_DISAMBIGUATION_FOR-_KANNADA_LANGUAGE_USING_DEEP_LEARNING
This is an repo that contains all the files developed while building the major project of final year.
# Kannada Word Sense Disambiguation (WSD) using Deep Learning

A lightweight, vocabulary-free **Two-Tower Neural Sense Ranker** that resolves lexical ambiguity in Kannada text by scoring sentence contexts against candidate sense definitions (glosses) from **Kannada IndoWordNet**. 

Developed by Janhavi Vijaykumar Gunaki, Gouramma Halangali, Ashwini Prakash Jangamshetti, Soumya Solapur, and Prof. Gurudevi Koluragi (Department of CSE - AI & ML, BLDEA’s V. P. Dr. P. G. Halakatti College of Engineering and Technology, Vijayapura).

---

## 🌟 Key Highlights

* **Zero Manual Annotation:** Automatically generates **8,043 weakly labelled training instances** across 4,452 lemmas directly from Kannada IndoWordNet synsets and example sentences.
* **Akshara-Aware Lemmatiser:** Uses a lexicon-driven prefix-matching lemmatiser that respects Kannada syllable (*akshara*) boundaries, achieving **92.5% accuracy** (37/40 target words) without requiring a full morphological analyser.
* **Vocabulary-Free & Ultra-Lightweight:** Uses CRC32 feature hashing (32,768 buckets) over surface forms, lemmas, and character 3- and 4-grams. The resulting model has **2.11M parameters (~8 MB)**, trains in under a minute on Google Colab, and runs smoothly on an ordinary laptop.
* **Probabilistic Ranking:** Outputs ranked candidate senses with calibrated probabilities via softmax over dot-product similarity scores.

---

## 🏗️ System Architecture

1. **Pre-processing:** Applies Unicode NFC normalization, tokenizes over the Kannada Unicode block, and extracts a ± 5-token context window around the target word.
2. **Feature Hashing:** Hashes surface forms, lemmatised tokens, and character n-grams into a shared `32,768 × 64` embedding bag.
3. **Two-Tower Network:**
   * **Context Tower:** Maps the context window features through two linear layers (`64 → 64`) with `tanh` activation and dropout into a 64-d context vector (c).
   * **Gloss Tower:** Maps candidate sense definitions and synonym lemmas (up to K ≤ 12 senses, excluding example sentences to prevent data leakage) into 64-d sense vectors (\(g_k\)).
4. **Scoring & Output:** Computes dot products (\(c \cdot g_k\)) and applies `softmax` to return ranked sense probabilities in a Jupyter Notebook or Gradio web interface.

---

## 📊 Performance Benchmarks

Evaluated on a strict **lemma-grouped split** (5,645 train / 1,222 validation / 1,176 test instances across **668 unseen lemmas**) and a hand-annotated gold set:

| Method | Test Top-1 (%) | Test Top-2 (%) | Gold Set Top-1 (%) |
| :--- | :---: | :---: | :---: |
| **Random Choice** (mean of 1/K) | 33.7% | — | 18.9% |
| **First-Listed Sense** | 32.0% | — | 19.4% |
| **Proposed Two-Tower System** | **36.3%** | **69.9%** | **32.3%** |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3
* **Deep Learning:** PyTorch (trained with AdamW optimizer and early stopping)
* **Lexicon:** Kannada IndoWordNet via `pyiwn` (22,042 synsets, 61,677 lemma strings)
* **Web UI / Deployment:** Gradio / Jupyter Notebook

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install torch pyiwn gradio
```

### 2. Project Structure: Jupyter source files which can be converted to python executable files 
```text
├── WSD_kannada.ipynb      # Model training file which also contains accuracy and loss graphs      
├──ALL_WSD_functions.ipynb # Contain all the functions like lemmatizer, tokenizer, cleaning data which will be used in gui
├──WSD_GUI.ipynb      # GUI which contains the UI and model deployment
├──Lemmatized_kannada_words.csv # data set which contains sentence and target words
├──annotation_templete.csv  # annotations developed during training
└──gold_senseID.csv    # gold sense ids for the words from the dataset
```
Note: All the Jupyter source files need to be converted to the python executable files to deploy and run the project functions
### 3. Running the Application
```bash
python app.py
```
Enter a Kannada sentence along with the ambiguous target word (and optional lemma) to view the top predicted IndoWordNet senses and their probability scores.

---

## 🔮 Future Scope

* Integrating pre-trained multilingual transformer encoders (e.g., IndicBERT, MuRIL) for richer contextual representations.
* Incorporating a dedicated Kannada morphological analyser.
* Expanding the human-annotated gold evaluation dataset and exposing the ranker as a REST API / browser plug-in.

