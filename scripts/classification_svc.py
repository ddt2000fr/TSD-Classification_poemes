import os
import re
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def normaliser_poeme(texte):
    # 1. Mise en minuscules (Majuscules -> minuscules)
    # Important pour que 'Soleil' et 'soleil' soient le même mot
    texte = texte.lower()

    # 2. Remplacer la ponctuation typographique par des espaces
    # On garde les accents français (àéèêëîïôûùç)
    # On remplace tout ce qui n'est pas une lettre accentuée ou un chiffre
    texte = re.sub(r'[^a-z0-9àéèêëîïôûùç]', ' ', texte)

    # 3. Supprimer les chiffres (souvent inutiles en poésie pour le style)
    texte = re.sub(r'\d+', '', texte)

    # 4. Nettoyage des espaces doubles ou triples
    texte = re.sub(r'\s+', ' ', texte).strip()

    return texte
# 1. Chargement des données
# On suppose que le dossier 'corpus' contient les 5 sous-dossiers d'auteurs
data = load_files('../corpus', encoding='utf-8', decode_error='replace')
X, y = data.data, data.target

# 2. Division Entraînement / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Vectorisation (Conversion du texte en nombres)
# On garde les accents, on retire les mots trop fréquents (stop_words)
vectorizer = TfidfVectorizer(
    preprocessor=normaliser_poeme, 
    stop_words=None # En poésie, même les petits mots (le, la) ont un style !
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Création et entraînement du SVM
# Le noyau 'linear' est souvent le meilleur pour le texte
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train_tfidf, y_train)

# 5. Prédictions et Évaluation
y_pred = clf.predict(X_test_tfidf)

#5. Création et affichage de la Matrice de Confusion
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)

disp.plot(cmap='Purples', ax=ax, xticks_rotation=45)
plt.title("Matrice de Confusion : Classification des Poètes")
plt.savefig("../resultats/confusion_matrix_SVC.png") # Sauvegarde l'image en PNG
plt.show()

print(f"Précision globale (Accuracy) : {accuracy_score(y_test, y_pred):.2%}")
print("\n--- Rapport détaillé ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))
