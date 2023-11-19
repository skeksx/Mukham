import os
import cv2
import threading
import numpy as np
from tqdm import tqdm
import concurrent.futures
from dataclasses import dataclass
import swap_mukham.default_paths as dp
from swap_mukham.utils import io as io_util
from swap_mukham.models.gender_age import GenderAge
from swap_mukham.models.retinaface import RetinaFace

cache = {}

@dataclass
class Face:
    bbox: np.ndarray = None
    kps: np.ndarray = None
    det_score: float = None
    embedding: np.ndarray = None
    gender: float = None
    age: float = None
    crop_112: np.ndarray = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )


def get_face_from_safetensor(path):
    from safetensors import safe_open
    face = None
    with safe_open(path, framework="pt", device="cpu") as f:
        if "embedding" in f.keys():
            embedding = f.get_tensor("embedding").numpy()
            face = Face(embedding=embedding)
    return face


def get_single_face(faces, method="best detection", center=None):
    method = method.lower()
    total_faces = len(faces)
    if total_faces == 0:
        return None
    if total_faces == 1:
        return faces[0]
    if method == "first detected":
        return faces[0]
    elif method == "best detection":
        return sorted(faces, key=lambda face: face["det_score"])[-1]
    elif method == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif method == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif method == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif method == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif method == "biggest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),)[-1]
    elif method == "smallest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),)[0]
    elif method == "center most":
        if center is None:
            raise ValueError("Image center is None!")
        return sorted(faces, key=lambda face: np.sqrt(((face['bbox'][0] + face['bbox'][2] // 2) - center[0]) ** 2 + ((face['bbox'][1] + face['bbox'][3] // 2) - center[1]) ** 2))[0]


def filter_face_by_age(faces, age_group):
    age_group = age_group.lower()
    if age_group == "child":
        return [face for face in faces if face["age"] <= 12]
    elif age_group == "teen":
        return [face for face in faces if face["age"] >= 13 and face["age"] <= 19]
    elif age_group == "adult":
        return [face for face in faces if face["age"] >= 20 and face["age"] <= 59]
    elif age_group == "senior":
        return [face for face in faces if face["age"] >= 60]
    elif age_group == "youngest":
        return [sorted(faces, key=lambda face: face["age"])[0]]
    elif age_group == "eldest":
        return [sorted(faces, key=lambda face: face["age"])[-1]]
    else:
        return faces


def filter_face_by_gender(faces, gender):
    gender = gender.lower()
    if gender == "male":
        return [face for face in faces if face["gender"] == 1]
    elif gender == "female":
        return [face for face in faces if face["gender"] == 0]
    else:
        return faces


def cosine_distance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 1 - np.dot(a, b)


def is_similar_face(face1, face2, threshold=0.6):
    distance = cosine_distance(face1["embedding"], face2["embedding"])
    return distance < threshold


class AnalyseFace:
    def __init__(self, **kwargs):
        self.detector = RetinaFace(model_file=dp.RETINAFACE_PATH, **kwargs)
        self.gender_age = GenderAge(model_file=dp.GENDERAGE_PATH, **kwargs)
        self.recognizer = None # load with swapper
        self.detection_size = (640, 640)
        self.detection_threshold = 0.5
        self.forced_detection = False

    def analyse(self, img, skip_task=[], top_fill=0):
        if self.recognizer is None and "embedding" not in skip_task:
            raise AttributeError("Recognizer not initialized")

        if not self.forced_detection:
            bboxes, kpss = self.detector.detect(img, input_size=self.detection_size, det_thresh=self.detection_threshold)
        else:
            threshholds = np.linspace(0.1, 0.9, 5, dtype=np.float32)
            bboxes, kpss = [], []
            for threshold in threshholds:
                b, k = self.detector.detect(
                    img, input_size=self.detection_size, det_thresh=threshold
                )
                if len(b) > len(bboxes):
                    bboxes, kpss = b, k

        faces = []
        for i in range(bboxes.shape[0]):
            if kpss is None:
                continue
            feat, gender, age, kps, crop_112 = None, None, None, None, None
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i]
            if "embedding" not in skip_task:
                feat, crop_112 = self.recognizer.get(img, kps, top_fill=top_fill)
            if "gender_age" not in skip_task:
                gender, age = self.gender_age.predict(img, kps)
            face = Face(
                bbox=bbox,
                kps=kps,
                det_score=det_score,
                embedding=feat,
                gender=gender,
                age=age,
                crop_112=crop_112,
            )
            faces.append(face)
        return faces

    def get_faces(self, image, **kwargs):
        if isinstance(image, str):
            image = cv2.imread(image)
        faces = self.analyse(image, **kwargs)
        return faces

    def get_face(self, image, method="center most", **kwargs):
        if isinstance(image, str):
            image = cv2.imread(image)
        height, width, _ = image.shape
        center = (width // 2, height // 2)
        faces = self.get_faces(image, **kwargs)
        face = get_single_face(faces, method=method, center=center)
        return face

    def get_face_averaged(self, image_or_path_list, base_face_index=0, method="mean", **kwargs):
        if not isinstance(image_or_path_list, list):
            image_or_path_list = [image_or_path_list]
        faces = []
        for image_or_path in image_or_path_list:
            if isinstance(image_or_path, np.ndarray):
                face = self.get_face(image_or_path, **kwargs)
                faces.append(face)
            elif isinstance(image_or_path, str):
                if io_util.is_image(image_or_path):
                    face = self.get_face(image_or_path, **kwargs)
                    faces.append(face)
                elif io_util.is_safetensor(image_or_path):
                    face = get_face_from_safetensor(image_or_path)
                    faces.append(face)
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return faces[0]
        base_face = faces[base_face_index]
        embeddings = [face["embedding"] for face in faces]
        if method == "mean":
            base_face["embedding"] = np.mean(embeddings, axis=0)
        elif method == "median":
            base_face["embedding"] = np.median(embeddings, axis=0)
        return base_face
