# %%
from pathlib import Path
from utils import *
tokens = get_N_tokens(N=3)
len(tokens)

# %%
tails = Path("./data/interim/").glob("*wac_tail_pp")
texts = list()
labels = list()
for file in tails:
    import re
    label = re.findall(pattern=r".+/([a-z]+)wac_tail_pp", string=str(file))[0]
    new_texts = read_and_split_file(str(file))
    texts.extend(new_texts)
    labels.extend([label for i in new_texts])
    
import pandas as pd
train_df = pd.DataFrame(data={
    "labels": labels,
    "text": texts
}).sample(frac=1)

    

# %%
SETimes = Path("data/interim/").glob("SETimes_[t,d]*.fasttext")
setimes_df = pd.concat([
    load_fasttext(str(i)) for i in SETimes
]).sample(frac=1)

# %%
twitter_paths = Path("data/interim/").glob("Twitter*.fasttext")
twitter_df = pd.concat([
    load_fasttext(str(i)) for i in twitter_paths
]).sample(frac=1)

# %%

def get_stats(N: int, train_df: pd.DataFrame, 
              eval_df: pd.DataFrame,
              classifier_type: str = "LinearSVC",
              ) -> dict:

    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        vocabulary=get_N_tokens(N), lowercase=True, binary=True)

    train_vectors = vectorizer.fit_transform(train_df.text)
    train_labels = train_df.labels
    if classifier_type == "LinearSVC":
        clf = LinearSVC(dual=False)
    elif classifier_type == "NaiveBayes":
        clf = GaussianNB()
    else:
        raise AttributeError(f"Got weird classifier_type: {classifier_type}, expected either LinearSVC or NaiveBayes")
    clf.fit(train_vectors.toarray(), train_labels)
    def evaluate(vectorizer, clf, eval_df):
        test_vectors = vectorizer.fit_transform(eval_df.text)
        y_true = eval_df.labels
        y_pred = clf.predict(test_vectors.toarray())
        from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
        LABELS = ["hr", "bs", "sr",  "me"]

        macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
        micro = f1_score(y_true, y_pred, labels=LABELS,  average="micro")
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        return {
            "N": N,
            "microF1": micro,
            "macroF1": macro,
            "accuracy": acc,
            "cm": cm.tolist(),
            # "y_true": y_true.tolist(),
            # "y_pred": y_pred.tolist(),
            "classifier": str(type(clf))
        }
    return evaluate(vectorizer, clf, eval_df)


# %%
import numpy as np
Ns = np.logspace(1.3, 2.3, 10)
results = list()
for i in range(5):
    for N in Ns:
        for clf_type in "NaiveBayes LinearSVC".split():
            N = int(N)
            d = get_stats(N, train_df, twitter_df, classifier_type=clf_type)
            d["eval_dataset"] = "Twitter"
            results.append(d)
            d = get_stats(N, train_df, setimes_df, classifier_type=clf_type)
            d["eval_dataset"] = "SETimes"
            results.append(d)
            import json
            with open("003_results2.json", "w") as f:
                json.dump(results, f)
