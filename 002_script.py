# %%
import os
import parse
import fasttext
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from typing import Union, List, Tuple
def get_labels_text(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Reads fasttext formatted file and extracts labels and text

    Args:
        path (Union[str, Path]): file, each line being a document. 
                                Line should start with __label__XX for label XX

    Returns:
        Tuple[List[str], List[str]]: Labels, texts
    """    
    labels, texts = list(), list()
    with open(str(path),"r") as f:
        pattern = "__label__{language} {text}"
        p = parse.compile(pattern)
        for line in f.readlines():
            rez = p.parse(line)
            labels.append(rez["language"])
            texts.append(rez["text"].replace("\n", " "))
    return labels, texts


from sklearn.metrics import accuracy_score, f1_score, classification_report


# %%

# %%
results = {
    "train on SETimes, test on SETimes": {
        "train": str(Path("data/interim/SETimes_train.fasttext")),
        "test": str(Path("data/interim/SETimes_test.fasttext"))
    },
    "train on SETimes, test on Twitter": {
        "train": str(Path("data/interim/SETimes_train.fasttext")),
        "test": str(Path("data/interim/Twitter_test.fasttext"))
    },
    "train on Twitter, test on SETimes":{
        "train": str(Path("data/interim/Twitter_train.fasttext")),
        "test": str(Path("data/interim/SETimes_test.fasttext"))
    },
    "train on Twitter, test on Twitter": {
        "train": str(Path("data/interim/Twitter_train.fasttext")),
        "test": str(Path("data/interim/Twitter_test.fasttext"))
    }
}
for setup in results:
    train = results[setup]["train"]
    test = results[setup]["test"]
    dev = results[setup]["train"].replace("train", "dev")
    results[setup]["runs"] = []
    for i in range(5):
        model = fasttext.train_supervised(input=train, 
                                          autotuneValidationFile=dev,
                                          autotuneDuration=600,
                                          maxn=10
                                    )
        y_true, texts = get_labels_text(test)
        y_pred =  [i[0].replace("__label__", "") for i in model.predict(texts)[0]]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels = "hr bs sr".split(),  average="macro")
        results[setup]["runs"].append({
            "accuracy": accuracy,
            "macroF1": f1,
            "y_true": y_true,
            "y_pred": y_pred
        })
        with open("002_fasttext_results.json", "w") as f:
            import json
            json.dump(results, f)

