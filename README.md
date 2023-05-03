# scikit-prodigy

Helpers to leverage scikit-learn pipelines in Prodigy.

## Recipes 

### `textcat.sklearn.binary`

This recipe assumes binary text classification done via scikit-learn. 
You're able to annotate as you would normally, but you can also set the
`--correct` flag which will train a scikit-learn model just before annotation.
You can then annotate more positive, negative or uncertain examples based 
on the `--prefer` setting in the recipe.

The default usage, which you should use to start with is:

```
python -m prodigy textcat.sklearn sklearn-demo examples.jsonl --label insult -F recipes/binary_textcat.py
```

Then, once we have positive/negative examples that sklearn could train on, you can
use it for model-in-the-loop annotation. 

```
python -m prodigy textcat.sklearn sklearn-demo examples.jsonl --label insult --correct --prefer uncertain -F recipes/binary_textcat.py
```

Feel free to take this recipe as a starting point to customise further!