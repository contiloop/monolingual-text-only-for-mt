# project/src/eval_metrics.py

class FinancialMetrics:
    def __init__(self):
        pass
    
    def compute(self, predictions, references):
        return {
            "bleu": 0.0,
            "number_accuracy": 0.0
        }
