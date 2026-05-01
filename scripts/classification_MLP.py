import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier # Nouveau modèle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

def normaliser_poeme(texte):
    texte = texte.lower()
    texte = re.sub(r'[^a-zàéèêëîïôûùç]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

# 1. Chargement sélectif (fichiers commençant par "poem")
corpus_path = '/Users/dariatupikina/Documents/Traitement_statistique/TSD-Classification_poemes/corpus'
authors = ['baudelaire', 'gautier', 'hugo', 'rimbaud', 'verlaine']
X, y = [], []

for idx, author in enumerate(authors):
    author_folder = os.path.join(corpus_path, author)
    if os.path.isdir(author_folder):
        for filename in os.listdir(author_folder):
            if filename.startswith("poem") and filename.endswith(".txt"):
                with open(os.path.join(author_folder, filename), 'r', encoding='utf-8') as f:
                    X.append(f.read())
                    y.append(idx)

# 2. Préparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Vectorisation
vectorizer = TfidfVectorizer(preprocessor=normaliser_poeme)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Configuration du MLPClassifier
# hidden_layer_sizes=(100,) signifie une couche cachée de 100 neurones
# max_iter=500 permet au réseau d'avoir assez de temps pour apprendre
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train_tfidf, y_train)

# 5. Évaluation
y_pred = mlp.predict(X_test_tfidf)

score_global = accuracy_score(y_test, y_pred)
print(f"L'exactitude globale (Accuracy) du réseau de neurones est de : {score_global:.2%}")

# 6. Matrice de Confusion
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=authors)
disp.plot(cmap='Blues', ax=ax)
plt.title("Matrice de Confusion : MLPClassifier (Réseau de Neurones)")
plt.savefig("/Users/dariatupikina/Documents/Traitement_statistique/TSD-Classification_poemes/resultats/confusion_matrix_MLP.png")
plt.show()

print(classification_report(y_test, y_pred, target_names=authors))