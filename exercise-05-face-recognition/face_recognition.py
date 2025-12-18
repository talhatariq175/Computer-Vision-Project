import os
import pickle

import cv2
import numpy as np

from cvproj_exc.config import Config


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding) #L2 normalization

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=15, max_distance=0.85, min_prob=0.6):
        # TODO: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        # 1. Get the embedding from the original color face image.
        color_embedding = self.facenet.predict(face)

        # 2. Convert the face to grayscale.
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 3. Convert the grayscale image back to 3 channels, as FaceNet requires it.
        gray_face_3_channel = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

        # 4. Get the embedding from the grayscale version.
        gray_embedding = self.facenet.predict(gray_face_3_channel)

        # 5. Add BOTH embeddings to the gallery.
        self.embeddings = np.vstack([self.embeddings, color_embedding, gray_embedding])

        # 6. IMPORTANT: Add the label twice, once for each embedding, to keep them aligned.
        self.labels.append(label)
        self.labels.append(label)

    # TODO: Predict the identity for a new face.
    def predict(self, face):
        # This is the advanced version of predict that uses both color and grayscale embeddings.

        # Return immediately if the gallery is empty.
        if len(self.labels) == 0:
            return "unknown", 0.0, float('inf')

        # Get Color Prediction
        color_embedding = self.facenet.predict(face)
        color_distances = np.linalg.norm(self.embeddings - color_embedding, axis=1)
        k_indices_color = np.argsort(color_distances)[:self.num_neighbours]
        k_labels_color = [self.labels[i] for i in k_indices_color]
        unique_labels_c, counts_c = np.unique(k_labels_color, return_counts=True)
        pred_label_c = unique_labels_c[np.argmax(counts_c)]

        # Get Grayscale Prediction
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face_3_channel = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
        gray_embedding = self.facenet.predict(gray_face_3_channel)
        gray_distances = np.linalg.norm(self.embeddings - gray_embedding, axis=1)
        k_indices_gray = np.argsort(gray_distances)[:self.num_neighbours]
        k_labels_gray = [self.labels[i] for i in k_indices_gray]
        unique_labels_g, counts_g = np.unique(k_labels_gray, return_counts=True)
        pred_label_g = unique_labels_g[np.argmax(counts_g)]

        # Combine and Make Final Decision
        # If both color and grayscale predictions agree on the same person...
        if pred_label_c == pred_label_g and pred_label_c != "unknown":
            predicted_label = pred_label_c

            # Find the distance to the prediction in both domains
            pred_class_indices = [i for i, label in enumerate(self.labels) if label == predicted_label]
            dist_c = np.min(color_distances[pred_class_indices])
            dist_g = np.min(gray_distances[pred_class_indices])

            # Average the distances and probabilities for a more robust decision
            dist_to_prediction = (dist_c + dist_g) / 2.0

            ki_c = np.max(counts_c)
            prob_c = ki_c / self.num_neighbours
            ki_g = np.max(counts_g)
            prob_g = ki_g / self.num_neighbours
            posterior_prob = (prob_c + prob_g) / 2.0

            # Use the averaged scores for the final open-set decision
            if dist_to_prediction > self.max_distance or posterior_prob < self.min_prob:
                return "unknown", posterior_prob, dist_to_prediction

            return predicted_label, posterior_prob, dist_to_prediction

        # If the predictions disagree, the result is ambiguous, so we default to "unknown".
        else:
            return "unknown", 0.0, float('inf')


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=5, max_iter=25): # Set num_cluster=5 as we trained model on 5 people
        # TODO: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # TODO
    def partial_fit(self, face):
        # 1. Use FaceNet to get the embedding for the new face.
        embedding = self.facenet.predict(face)

        # 2. Add the embedding to our list of embeddings to be clustered later.
        self.embeddings = np.vstack([self.embeddings, embedding])

    # TODO
    def fit(self):
        # 1. Initialize cluster centers by randomly choosing 'k' points from the dataset.
        initial_indices = np.random.choice(self.embeddings.shape[0], self.num_clusters, replace=False)
        self.cluster_center = self.embeddings[initial_indices]

        # 2. Start the main k-means loop.
        for _ in range(self.max_iter):
            # 3. ASSIGNMENT STEP:
            #    Calculate the distance from each embedding to every cluster center.
            #    The result is a matrix where rows are embeddings and columns are cluster centers.
            distances = np.linalg.norm(self.embeddings[:, np.newaxis] - self.cluster_center, axis=2)

            #    Assign each embedding to the closest cluster center (the one with the minimum distance).
            self.cluster_membership = np.argmin(distances, axis=1)

            # 4. UPDATE STEP:
            #    Calculate the new cluster centers by taking the mean of all embeddings
            #    assigned to each cluster.
            new_centers = np.array([self.embeddings[self.cluster_membership == i].mean(axis=0)
                                    for i in range(self.num_clusters)])

            # 5. Check for convergence: if the cluster centers haven't changed, we can stop early.
            if np.allclose(self.cluster_center, new_centers):
                break

            self.cluster_center = new_centers

    # TODO
    def predict(self, face):
        # 1. Use FaceNet to get the embedding for the new face.
        embedding = self.facenet.predict(face)

        # 2. Calculate the distances from the new embedding to each of the final cluster centers.
        distances_to_clusters = np.linalg.norm(self.cluster_center - embedding, axis=1)

        # 3. The best matching cluster is the one with the smallest distance.
        predicted_label = np.argmin(distances_to_clusters)

        # 4. Return the predicted cluster label and the list of distances to all clusters.
        return predicted_label, distances_to_clusters
