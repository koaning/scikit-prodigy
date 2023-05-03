class ProdigyEstimator:
    def __init__(self, estimator, label=None):
        self.estimator = estimator
    
    def __call__(self, examples):
        probas = self.estimator.predict_proba([ex['text'] for ex in examples])
        preds = self.estimator.predict([ex['text'] for ex in examples])
        for prob, pred in zip(probas, preds):
            yield (float(prob.max()), pred)
    
    def fit_from_prodigy(self, dataset):
        print(dataset)
    
    def update(examples):
        pass


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import CountVectorizer

pipe = make_pipeline(CountVectorizer(), LogisticRegression())