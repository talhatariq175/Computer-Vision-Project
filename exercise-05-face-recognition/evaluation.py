import pickle

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    #TODO
    def run(self):
        # 1. Train the classifier on the training data.
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # 2. Get the initial predictions and similarity scores for the test data.
        pred_labels, pred_similarities = self.classifier.predict_labels_and_similarities(
            self.test_embeddings
        )

        identification_rates = []
        similarity_thresholds = []

        # 3. For each False Alarm Rate we want to test...
        for far in self.false_alarm_rate_range:
            # 4. Find the similarity threshold that produces this FAR.
            threshold = self.select_similarity_threshold(pred_similarities, far)
            similarity_thresholds.append(threshold)

            # 5. Apply the threshold to the predictions. Anything below it becomes 'unknown'.
            final_predictions = np.copy(pred_labels)
            final_predictions[pred_similarities < threshold] = UNKNOWN_LABEL

            # 6. Calculate the identification rate using these new thresholded predictions.
            id_rate = self.calc_identification_rate(final_predictions)
            identification_rates.append(id_rate)

        # Report all performance measures.
        evaluation_results = {
            "similarity_thresholds": np.array(similarity_thresholds),
            "identification_rates": np.array(identification_rates),
        }

        return evaluation_results
    #TODO
    def select_similarity_threshold(self, similarity, false_alarm_rate):
        # 1. Get the similarity scores for ONLY the unknown subjects (label == -1).
        unknown_similarities = similarity[self.test_labels == UNKNOWN_LABEL]

        # 2. To achieve a False Alarm Rate of 1% (0.01), we need to find a
        #    threshold that rejects 99% of the unknowns. This threshold is the
        #    99th percentile of the unknown similarity scores.
        percentile = 100 * (1 - false_alarm_rate)
        return np.percentile(unknown_similarities, percentile)

    #TODO
    def calc_identification_rate(self, prediction_labels):
        # 1. Find the ground truth labels for only the known subjects (label != -1).
        known_indices = np.where(self.test_labels != UNKNOWN_LABEL)
        known_gt_labels = self.test_labels[known_indices]

        # 2. Get the predicted labels for those same subjects.
        known_pred_labels = prediction_labels[known_indices]

        # 3. Count how many of your predictions match the ground truth labels.
        num_correct = np.sum(known_pred_labels == known_gt_labels)

        # 4. The rate is the number of correct guesses divided by the total number of known people.
        if len(known_gt_labels) == 0:
            return 0.0
        return num_correct / len(known_gt_labels)
