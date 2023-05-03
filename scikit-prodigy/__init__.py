class ProdigyEstimator:
    def __init__(self, estimator, dataset, label):
        self.estimator = estimator
        self.dataset = dataset
        self.label = label
    
    def __call__(self, examples):
        probas = self.estimator.predict_proba([ex['text'] for ex in examples])
        preds = self.estimator.predict([ex['text'] for ex in examples])
        for prob, pred in zip(probas, preds):
            yield (float(prob.max()), pred)
    
    def fit_from_prodigy(self):
        """This only works for binary classes now"""
        db = connect()
        examples = db.get_dataset_examples(self.dataset)
        examples = [(ex, ex['answer'] == 'accept') 
                    for ex in examples 
                    if ex['answer'] in ['accept', 'reject']]
        texts, labels = zip(*examples)
        self.estimator.fit(texts, labels)
        msg.info(f"Trained scikit-learn pipeline on {len(examples)} examples.")
        return self
    
    def update(examples):
        examples = [(ex, ex['answer'] == 'accept') 
                    for ex in examples 
                    if ex['answer'] in ['accept', 'reject']]
        texts, labels = zip(*examples)
        self.estimator.partial_fit(texts, labels)
        return self

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import CountVectorizer

pipe = make_pipeline(CountVectorizer(), LogisticRegression())
est = ProdigyEstimator(pipe, dataset="name", label="pos")