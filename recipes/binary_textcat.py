"""
This recipe assumes binary text classification done via scikit-learn. 
You're able to annotate as you would normally, but you can also set the
`--correct` flag which will train a scikit-learn model just before annotation.
You can then annotate more positive, negative or uncertain examples based 
on the `--prefer` setting in the recipe.

## USAGE

The default usage, which you should use to start with. 

```
python -m prodigy textcat.sklearn sklearn-demo examples.jsonl --label insult -F recipe.py
```

Then, once we have positive/negative examples that sklearn could train on, you can
use it for model-in-the-loop annotation. 

```
python -m prodigy textcat.sklearn sklearn-demo examples.jsonl --label insult --correct --prefer uncertain -F recipe.py
```
"""

from prodigy import get_stream
from prodigy.components.db import connect

from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer

import prodigy
from prodigy.util import msg
from prodigy.components.sorters import prefer_uncertain, prefer_high_scores, prefer_low_scores


def fit_from_prodigy(estimator, dataset, label):
    db = connect()
    examples = db.get_dataset_examples(dataset)
    examples = [(ex, ex['answer'] == 'accept') 
                for ex in examples 
                if ex['answer'] in ['accept', 'reject']]
    texts, labels = zip(*examples)
    estimator.fit(texts, labels)
    msg.info(f"Trained scikit-learn pipeline on {len(examples)} examples.")
    return estimator


@prodigy.recipe(
    "textcat.sklearn.binary",
    dataset=("Dataset to save answers to", "positional", None, str),
    path=("Path to text data", "positional", None, str),
    correct=("Correct a scikit-learn model trained on the data", "flag", "c", bool),
    prefer=("When correcting, what do we prefer", "option", "p", str),
    label=("Pass comma seperated labels to add to dataset", "option", "l", str)
)
def my_custom_recipe(dataset, path, label, correct=False, prefer="uncertain"):
    if not label:
        msg.fail("You forgot to pass --label", exits=True)
    
    # Load your own streams from anywhere you want
    stream = get_stream(path)

    def add_options(stream, label):
        for task in stream:
            task['label'] = label
            yield task

    def add_preds(examples, pipe):
        for ex in examples:
            pred = pipe.predict([ex])[0]
            proba = pipe.predict_proba([ex])[0]
            # This is the proba value for the `True` class
            score = float(proba[pipe.classes_ == True])
            new_example = {
                **ex, 
                "label": label, 
                "meta": {"score": score}
            }
            yield (score, new_example)
    
    def filter_by_preference(stream, preference):
        msg.info(f"Using model in the loop with {preference=}.")
        if preference == "uncertain":
            return prefer_uncertain(stream)
        if preference == "high":
            return prefer_high_scores(stream)
        if preference == "low":
            return prefer_low_scores(stream)
        raise ValueError("Preference must be either `uncertain`, `low` or `high`.")
    
    stream = add_options(stream, label)

    if correct:
        # Note the important `class_weight="balanced"` there!
        pipe = make_pipeline(
            FunctionTransformer(lambda d: [ex['text'] for ex in d]),
            CountVectorizer(), 
            LogisticRegression(class_weight="balanced")
        )

        pipe = fit_from_prodigy(pipe, dataset, label)
        stream = filter_by_preference(add_preds(stream, pipe), prefer)


    return {
        "dataset": dataset,
        "view_id": "classification",
        "stream": stream,
    }
