import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from operator import itemgetter
from subprocess import call
import matplotlib.pyplot as plt

def load_data(real_filename, fake_filename):
    with open(real_filename) as f:
        real_data = f.readlines()
    with open(fake_filename) as f:
        fake_data = f.readlines()
    data = real_data + fake_data
    labels = np.zeros(len(real_data) + len(fake_data))
    labels[:len(real_data)] = 1
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.3, random_state=2019)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=2019)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)
    vocab = vectorizer.get_feature_names()
    return X_train, y_train, X_val, y_val, X_test, y_test, vocab

def select_model(X_train, y_train, X_val, y_val, X_test, y_test):
    depths = [1, 3, 5, 10, 20]
    criteria = ["gini", "entropy"]
    models = []
    for depth in depths:
        for criterion in criteria:
            clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion)
            clf.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, clf.predict(X_val))
            models.append({
                "depth": depth,
                "criterion": criterion,
                "accuracy": accuracy,
                "model": clf
            })
            print(f"depth = {depth}, criterion = {criterion}, acc = {accuracy}")
    return models

def entropy(labels):
    _, frequencies = np.unique(labels, return_counts=True)
    p = frequencies / len(labels)
    H = np.sum([-x * np.log2(x) for x in p])
    return H

def conditional_entropy(X, Y):
    H_cond = 0
    X_unique, X_frequencies = np.unique(X, return_counts=True)
    p_X = X_frequencies / len(X)
    for i, x in enumerate(X_unique):
        for k, y in enumerate(np.unique(Y)):
            p_x = p_X[i]
            p_xy = ((X == x) & (Y == y)).sum() / len(X)
            if p_x == 0 or p_xy == 0:
                continue
            H_cond += p_xy * np.log(p_x / p_xy)
    return H_cond

def information_gain(X, Y):
    return entropy(Y) - conditional_entropy(X, Y)

def full_information_gain(X, Y):
    X = np.array(X.todense())
    igs = []
    for i in range(X.shape[1]):
        igs.append(information_gain(X[:, i].reshape(-1, 1), Y.reshape(-1, 1)))
    return igs

if __name__ == "__main__":
    data_dir = "data"
    real_file = "clean_real.txt"
    fake_file = "clean_fake.txt"
    X_train, y_train, X_val, y_val, X_test, y_test, vocab = \
        load_data(os.path.join(data_dir, real_file), os.path.join(data_dir, fake_file))
    models = select_model(X_train, y_train, X_val, y_val, X_test, y_test)
    models = sorted(models, key=itemgetter("accuracy"), reverse=True) 
    best_model = models[0]["model"]
    export_graphviz(best_model, out_file='tree.dot', 
                rounded = True, proportion = False, 
                precision = 2, filled = True, max_depth=1)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('tree.png'))
    plt.axis('off');
    plt.show();

    igs = np.array(full_information_gain(X_train, y_train))
    id_sorted = np.argsort(igs)[::-1]
    igs_sorted = igs[id_sorted]
    k = 10
    vocab = np.array(vocab)
    top_split_id = 4185
    print(f"top previous word: {vocab[top_split_id]},", 
          f"{information_gain(np.array(X_train.todense())[:, top_split_id], y_train)}")
    print("\n".join(map(lambda x: f"{x[0]}: {x[1]}", zip(vocab[id_sorted[:k]], igs_sorted))))