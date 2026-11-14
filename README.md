# Évaluation des tâches NLP : Classification, Sequence Labeling, Traduction automatique

Devoir – Master Data & IA (INPT), *ML Evaluation* 2026.

## Contenu
- `NLP_Evaluation.ipynb` : notebook avec une cellule d'exemple par tâche.
- `requirements.txt` : dépendances.

## Exécution
Recommandé : Google Colab (importer le notebook, puis *Exécuter tout*). En local :
```bash
pip install -r requirements.txt
jupyter notebook NLP_Evaluation.ipynb
```
Les cellules 2 et 3 téléchargent des modèles Hugging Face au premier lancement (connexion internet requise).

## Documentation des cellules
| Cellule | Tâche | Exemple | Métriques | Bibliothèques |
|---|---|---|---|---|
| 1 | Classification | Sentiment (TF-IDF + régression logistique) sur 300 phrases synthétiques avec 10 % de bruit d'annotation | Accuracy, Precision, Recall, F1, ROC-AUC, matrice de confusion | scikit-learn |
| 2 | Sequence labeling | NER avec `dslim/bert-base-NER` sur 5 phrases annotées à la main (BIO) | Accuracy token, Precision/Recall/F1 par entité, rapport par type | transformers, seqeval |
| 3 | Traduction automatique | EN→FR avec `Helsinki-NLP/opus-mt-en-fr` (chargé via `AutoModelForSeq2SeqLM`, compatible transformers v5) sur 5 phrases avec référence | BLEU, chrF, TER | transformers, sacrebleu |

### Détails importants
- **Cellule 2** : `to_bio()` convertit les entités prédites par le modèle (offsets de caractères) en étiquettes BIO alignées sur les mots, afin de comparer avec l'annotation de référence.
- **Cellule 3** : `sacrebleu` attend les références sous forme de liste de listes (`[references]`). Pour TER, une valeur plus basse est meilleure.
- Les jeux de données sont volontairement petits pour illustrer le calcul des métriques ; ils ne remplacent pas une évaluation sur un vrai benchmark (IMDB, CoNLL-2003, WMT).
