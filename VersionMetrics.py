class VersionMetrics:
    number_of_requests: int
    number_of_positive_predictions: int
    number_of_negative_predictions: int
    number_of_correct_predictions: int
    number_of_incorrect_predictions: int

    def __init__(self):
        self.number_of_requests = 0
        self.number_of_positive_predictions = 0
        self.number_of_negative_predictions = 0
        self.number_of_correct_predictions = 0
        self.number_of_incorrect_predictions = 0