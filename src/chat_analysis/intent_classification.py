from transformers import pipeline

class IntentClassification():
    """
    IntentClassification class.
    """

    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.candidate_labels = ['Information Seeking', 'Expressing Concern', 'Deliberation', 'Gratitude', 'Mentioning Previous Experience']

    def classify(self, text):
        """
        Classify the given text.
        """
        result = self.classifier(text, self.candidate_labels)
        score = max(result['scores'])
        label = result['labels'][result['scores'].index(score)]
        return label, score


if __name__ == "__main__":
    intent_classification = IntentClassification()
    print(intent_classification.classify("Hello, I'm interested in purchasing the new iPhone 14. Can you provide me with some information about it?"))