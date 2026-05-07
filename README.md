# Classification des textes de poèmes par auteur en français

Projet de classification automatique de poèmes français du XIXe siècle par auteur à l’aide de méthodes de machine learning supervisé.

## Auteurs

- Anna Sugak
- Daria Tupikina

---

# Description du projet

Ce projet a pour objectif d’identifier automatiquement l’auteur d’un poème parmi cinq grands poètes français du XIXe siècle :

- Charles Baudelaire
- Paul Verlaine
- Arthur Rimbaud
- Théophile Gautier
- Victor Hugo

Le projet repose sur des techniques de traitement automatique du langage naturel (NLP) et plusieurs algorithmes de classification supervisée :

- Naive Bayes
- SVM (LinearSVC)
- Réseau de neurones (MLPClassifier)

---

# Objectifs

- Construire un corpus de poèmes français équilibré
- Prétraiter les textes littéraires
- Transformer les textes en vecteurs numériques (TF-IDF)
- Comparer plusieurs algorithmes de classification
- Évaluer les performances des modèles

---

# Corpus

## Source des données

Les textes proviennent de Project Gutenberg :

https://www.gutenberg.org

### Recueils utilisés

| Auteur | Œuvre |
|---|---|
| Charles Baudelaire | *Les Fleurs du Mal* |
| Paul Verlaine | *Œuvres complètes, Volume 1* |
| Arthur Rimbaud | *Œuvres : Vers et proses* |
| Théophile Gautier | *Poésies complètes, Tome 1* |
| Victor Hugo | *Les Contemplations* |

Tous les textes appartiennent au domaine public.

---

# Résumé du corpus

| Auteur | Poèmes extraits | Poèmes utilisés |
|---|---:|---:|
| Baudelaire | 111 | 111 |
| Verlaine | 217 | 111 |
| Rimbaud | 144 | 111 |
| Gautier | 247 | 111 |
| Hugo | 167 | 111 |
| **Total** | **886** | **555** |

Afin d’équilibrer les classes, 111 poèmes ont été sélectionnés aléatoirement pour chaque auteur.

---

# Prétraitement des données

Un script Python `split_poems.py` a été développé afin de :

- supprimer les en-têtes et pieds de page de Project Gutenberg ;
- détecter les titres des poèmes (lignes en majuscules) ;
- découper automatiquement les recueils en fichiers individuels ;
- organiser les textes par auteur.

Une fonction de normalisation a également été utilisée pour :

- convertir le texte en minuscules ;
- supprimer la ponctuation ;
- supprimer les chiffres romains et arabes.

---

# Méthodologie

## 1. Vectorisation TF-IDF

Les textes sont vectorisés avec `TfidfVectorizer` de scikit-learn :

- vocabulaire limité à 5 000 termes ;
- n-grammes `(1,2)` : unigrammes + bigrammes.

---

## 2. Séparation des données

- 80% entraînement
- 20% test
- stratification par auteur
- `random_state = 42`

---

# Modèles utilisés

## Multinomial Naive Bayes

| Paramètre | Valeur |
|---|---|
| Algorithme | MultinomialNB |
| Vectorisation | TF-IDF |
| Vocabulaire | 5000 termes |
| N-grammes | (1,2) |
| Train/Test | 80/20 |
| random_state | 42 |

---

## SVM — LinearSVC

| Paramètre | Valeur |
|---|---|
| Algorithme | LinearSVC |
| Vectorisation | TF-IDF |
| N-grammes | (1,2) |
| Kernel | linear |
| C | 1.0 |
| Train/Test | 80/20 |
| random_state | 42 |

Le paramètre `kernel` détermine la forme de la frontière de séparation entre les classes.

Le paramètre `C` contrôle la tolérance aux erreurs de classification.

---

## Réseau de neurones — MLPClassifier

| Paramètre | Valeur |
|---|---|
| Algorithme | MLPClassifier |
| Vectorisation | TF-IDF |
| N-grammes | (1,2) |
| hidden_layer_sizes | (100,) |
| max_iter | 500 |
| Train/Test | 80/20 |
| random_state | 42 |

- `hidden_layer_sizes` correspond au nombre de neurones de la couche cachée ;
- `max_iter` définit le nombre maximal d’itérations d’apprentissage.

---

# Résultats

## Accuracy globale

| Modèle | Accuracy |
|---|---:|
| Naive Bayes | 68% |
| LinearSVC | 74.11% |
| MLPClassifier | 81.98% |

---

# Résultats détaillés

## Naive Bayes

| Auteur | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Baudelaire | 0.74 | 0.77 | 0.76 |
| Verlaine | 0.83 | 0.68 | 0.75 |
| Rimbaud | 0.54 | 0.95 | 0.69 |
| Gautier | 0.61 | 0.61 | 0.61 |
| Hugo | 1.00 | 0.36 | 0.53 |

Accuracy : **0.68**

### Observations

- Victor Hugo possède une précision parfaite mais un rappel faible ;
- Rimbaud est souvent sur-prédit par le modèle ;
- Verlaine est l’auteur le plus difficile à identifier.

---

## LinearSVC

| Auteur | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Baudelaire | 0.72 | 0.67 | 0.69 |
| Gautier | 0.85 | 0.85 | 0.85 |
| Hugo | 0.89 | 0.76 | 0.82 |
| Rimbaud | 0.65 | 0.69 | 0.67 |
| Verlaine | 0.62 | 0.73 | 0.67 |

Accuracy : **74.11%**

### Observations

- Gautier et Hugo obtiennent les meilleurs scores ;
- Hugo présente un rappel très élevé ;
- Les styles de Rimbaud et Verlaine restent plus difficiles à distinguer.

---

## MLPClassifier

| Auteur | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Baudelaire | 0.92 | 0.73 | 0.81 |
| Gautier | 0.89 | 0.85 | 0.87 |
| Hugo | 0.65 | 1.00 | 0.79 |
| Rimbaud | 0.95 | 0.95 | 0.95 |
| Verlaine | 0.73 | 0.67 | 0.70 |

Accuracy : **81.98%**

### Observations

- Rimbaud et Gautier possèdent les styles les plus distinctifs ;
- Verlaine reste l’auteur le plus difficile à classifier ;
- Le réseau de neurones offre les meilleures performances globales.

---

# Technologies utilisées

- Python
- scikit-learn
- NumPy
- pandas
- TF-IDF
- Machine Learning supervisé


---

# Évaluation

Les métriques utilisées :

- Accuracy
- Precision
- Recall
- F1-score

Implémentation via `sklearn.metrics`.

---

# Conclusion

Les expérimentations montrent la hiérarchie suivante des modèles selon l’accuracy :

1. MLPClassifier — 81.98%
2. LinearSVC — 74.11%
3. Multinomial Naive Bayes — 68%

Le modèle MLPClassifier obtient les meilleurs résultats globaux grâce à sa capacité à capturer des relations non linéaires complexes entre les mots.

Le modèle SVM surpasse Naive Bayes pour presque tous les auteurs et identifie efficacement les frontières stylistiques entre les classes.

Ce projet montre que les techniques modernes de NLP et de machine learning permettent d’identifier efficacement le style littéraire d’un auteur à partir de ses poèmes.

---

# Références

- Project Gutenberg — https://www.gutenberg.org
- Documentation scikit-learn — https://scikit-learn.org/stable/
- Documentation Python — https://docs.python.org/3/
