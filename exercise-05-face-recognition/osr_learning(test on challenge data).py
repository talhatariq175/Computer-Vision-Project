from collections.abc import Callable

import numpy as np
import pandas as pd

from config import Config
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def spl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the single pseudo label (SPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    spl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'spl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    # We choose a Support Vector Classifier with a linear kernel.
    # It's efficient and the 'C' parameter can be tuned to prevent overfitting.
    # We set probability=True to get confidence scores later.
    clf = SVC(kernel='linear', probability=True, random_state=42, C=1.0)

    # Train the classifier on the provided data.
    clf.fit(x_train, y_train)

    def spl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.

        # Use the trained SVC to predict labels and get confidence scores.
        y_pred = clf.predict(x_test)
        y_score = np.max(clf.predict_proba(x_test), axis=1)

        return y_pred, y_score

    return spl_predict_fn


def mpl_training(
    x_train: np.ndarray, y_train: np.ndarray
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Implementation of the multi pseudo label (MPL) approach.
    Do NOT change the interface of this function. For benchmarking we expect the given inputs and
    return values. Introduce additional helper functions if desired.

    Parameters
    ----------
    x_train : array, shape (n_samples, n_features). The feature vectors for training.
    y_train : array, shape (n_samples,). The ground truth labels of samples x.

    Returns
    -------
    mpl_predict_fn :
        Callable, a function that holds a reference to your trained estimator and uses it to
        predict class labels and scores for the incoming test data.

        Parameters
        ----------
        x_test : array, shape (n_test_samples, n_features). The feature vectors for testing.

        Returns
        -------
        y_pred :    array, shape (n_samples,). The predicted class labels.
        y_score :   array, shape (n_samples,).
                    The similarities or confidence scores of the predicted class labels. We assume
                    that the scores are confidence/similarity values, i.e., a high value indicates
                    that the class prediction is trustworthy.
                    To be more precise:
                    - Returning probabilities in the range 0 to 1 is fine if 1 means high
                      confidence.
                    - Returning distances in the range -inf to 0 (or +inf) is fine if 0 (or +inf)
                      means high confidence.

                    Please ensure that your score is formatted accordingly.
    """

    # TODO: 1) Use arguments 'x_train' and 'y_train' to find and train a suitable estimator.
    #       2) Use your trained estimator within the function 'mpl_predict_fn' to predict class
    #          labels and scores for the incoming test data 'x_test'.

    # 1. Create a copy of the labels to modify for MPL.
    y_train_mpl = np.copy(y_train)
    max_known_label = int(np.max(y_train))
    kuc_indices = np.where(y_train == -1)[0]

    # 2. Create new, unique pseudo-labels for each unknown sample.
    num_kucs = len(kuc_indices)
    pseudo_labels = np.arange(max_known_label + 1, max_known_label + 1 + num_kucs)
    y_train_mpl[kuc_indices] = pseudo_labels

    # 3. Train the same efficient Linear SVC on the new MPL labels.
    clf = SVC(kernel='linear', probability=True, random_state=42, C=1.0)
    clf.fit(x_train, y_train_mpl)

    def mpl_predict_fn(x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: In this nested function, you can use everything you have trained in the outer
        #       function.

        # 4. Predict using the MPL-trained classifier.
        y_pred_mpl = clf.predict(x_test)
        y_score = np.max(clf.predict_proba(x_test), axis=1)

        # 5. Map all predicted pseudo-labels back to the standard unknown label (-1).
        y_pred = np.copy(y_pred_mpl)
        y_pred[y_pred > max_known_label] = -1

        return y_pred, y_score

    return mpl_predict_fn


def load_challenge_train_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the challenge training data.

    Returns
    -------
    x : array, shape (n_samples, n_features). The feature vectors.
    y : array, shape (n_samples,). The corresponding labels of samples x.
    """
    df = pd.read_csv(Config.CHAL_TRAIN_DATA, header=None).values
    x = df[:, :-1]
    y = df[:, -1].astype(int)
    return x, y


def main():
    # 1. Load the full dataset provided by the professor.
    x_full, y_full = load_challenge_train_data()

    # 2. Split the data into a 70% training set and a 30% validation set.
    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )

    # TODO: implement
    # Train the models using the 70% training portion.
    spl_predict_fn = spl_training(x_train, y_train)

    # TODO: implement
    mpl_predict_fn = mpl_training(x_train, y_train)

    # TODO: No todo, but this is roughly how we will test your implementation (with real data). So
    #       please make sure that this call (besides the unit tests) does what it is supposed to do.

    # We now test on our 30% validation set to get a meaningful accuracy score.
    print("\n--- Validation Results on 30% of the data ---")
    for name, predict_fn in (("SPL", spl_predict_fn), ("MPL", mpl_predict_fn)):
        y_pred, y_score = predict_fn(x_test)
        accuracy = np.equal(y_test, y_pred).sum() / len(y_test)
        print(f"Accuracy for {name}: {accuracy:.2%}")


if __name__ == "__main__":
    main()
