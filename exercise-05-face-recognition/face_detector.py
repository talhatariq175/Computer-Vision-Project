import cv2
import numpy as np
from mtcnn import MTCNN


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(self, tm_window_size=25, tm_threshold=0.2, aligned_image_size=224):
        # Prepare face alignment.
        self.detector = MTCNN()

        # Reference (initial face detection) for template matching.
        self.reference = None

        # Size of face image after landmark-based alignment.
        self.aligned_image_size = aligned_image_size

        # TODO: Specify all parameters for template matching.
        self.tm_window_size = tm_window_size
        self.tm_threshold = tm_threshold

    # TODO: Track a face in a new image using template matching.
    def track_face(self, image):
        # 1. If no face is being tracked yet, run a full detection to find one.
        # This initializes the tracker with a "reference" face.
        if self.reference is None:
            self.reference = self.detect_face(image)
            return self.reference

        # 2. Define a small search area around the last known face position.
        # This is faster than searching the whole image.
        ref_rect = self.reference["rect"]
        template = self.reference["aligned"]

        # Calculate the coordinates of the search window
        p_top = max(0, ref_rect[1] - self.tm_window_size)
        p_bottom = min(image.shape[0], ref_rect[1] + ref_rect[3] + self.tm_window_size)
        p_left = max(0, ref_rect[0] - self.tm_window_size)
        p_right = min(image.shape[1], ref_rect[0] + ref_rect[2] + self.tm_window_size)
        search_window = image[p_top:p_bottom, p_left:p_right]

        # 3. Use OpenCV's template matching to find the template in the search window.
        # We check if the template can fit inside the search window before matching.
        if template.shape[0] > search_window.shape[0] or template.shape[1] > search_window.shape[1]:
            # If not, the track is lost. Reset and re-detect.
            self.reference = None
            return self.track_face(image)

        res = cv2.matchTemplate(search_window, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        response = max_val

        # 4. Check if the match quality is too low (below the threshold).
        # If so, the track is lost. Reset the reference and run a full detection.
        if response < self.tm_threshold:
            self.reference = None
            return self.track_face(image)

        # 5. If the match is good, calculate the new face position.
        new_face_rect = (
            p_left + max_loc[0],
            p_top + max_loc[1],
            ref_rect[2],
            ref_rect[3],
        )

        # Align the newly found face and update the reference for the next frame.
        aligned = self.align_face(image, new_face_rect)
        self.reference = {
            "rect": new_face_rect,
            "image": image,
            "aligned": aligned,
            "response": response,
        }

        return self.reference

    # Face detection in a new image.
    def detect_face(self, image):
        # Retrieve all detectable faces in the given image.
        if not (
            detections := self.detector.detect_faces(image)
        ):
            self.reference = None
            return None

        # Select face with largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = detections[largest_detection]["box"]

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        return {"rect": face_rect, "image": image, "aligned": aligned, "response": 0}

    # Face alignment to predefined size.
    def align_face(self, image, face_rect):
        return cv2.resize(
            self.crop_face(image, face_rect),
            dsize=(self.aligned_image_size, self.aligned_image_size),
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image, face_rect):
        top = max(face_rect[1], 0)
        left = max(face_rect[0], 0)
        bottom = min(face_rect[1] + face_rect[3] - 1, image.shape[0] - 1)
        right = min(face_rect[0] + face_rect[2] - 1, image.shape[1] - 1)
        return image[top:bottom, left:right, :]
