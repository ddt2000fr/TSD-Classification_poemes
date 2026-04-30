import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)

def load_corpus(corpus_dir, max_per_author=111):
    texts = []
    labels = []
    authors = ['baudelaire', 'verlaine', 'rimbaud', 'gautier', 'hugo']
    
    for author in authors:
        author_dir = os.path.join(corpus_dir, author)
        filenames = sorted([f for f in os.listdir(author_dir)
                           if f.startswith('poeme_') and f.endswith('.txt')])
        filenames = random.sample(filenames, min(max_per_author, len(filenames)))
        
        for filename in filenames:
            filepath = os.path.join(author_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(author)
    
    return texts, labels

texts, labels = load_corpus('corpus', max_per_author=111)
print(f"Total: {len(texts)} poèmes")
print(f"Baudelaire: {labels.count('baudelaire')}")
print(f"Verlaine: {labels.count('verlaine')}")
print(f"Rimbaud: {labels.count('rimbaud')}")
print(f"Gautier: {labels.count('gautier')}")
print(f"Hugo: {labels.count('hugo')}")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\nTrain: {X_train.shape[0]} poèmes")
print(f"Test: {X_test.shape[0]} poèmes")

authors = ['baudelaire', 'verlaine', 'rimbaud', 'gautier', 'hugo']

print("\n=== Naive Bayes ===")
clf_nb = MultinomialNB()
clf_nb.fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)

print(classification_report(y_test, y_pred_nb, target_names=authors))

cm_nb = confusion_matrix(y_test, y_pred_nb, labels=authors)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d',
            xticklabels=authors,
            yticklabels=authors)
plt.title('Matrice de confusion - Naive Bayes')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('resultats/confusion_matrix_nb.png')
print("Matrice de confusion sauvegardée: resultats/confusion_matrix_nb.png")
