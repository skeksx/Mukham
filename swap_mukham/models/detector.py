import cv2
import sys
import onnx
import onnxruntime
import numpy as np
from tqdm import tqdm
from swap_mukham.utils.device import OnnxInferenceSession


def prepare_image(img):
    img = cv2.resize(img, (224, 224)).astype("float32")
    img -= np.array([104, 117, 123], dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img

class Detector(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def detect_image(self, image, threshold=0.9):
        if isinstance(image, str):
            image = cv2.imread(image)
        img = prepare_image(image)
        score = self.session.run(None, {self.input_name: img})[0][0][1]
        if score >= threshold:
            return False
        return True

    def detect_video(self, video_path, threshold=0.9, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(total_frames, max_frames)
        indexes = np.arange(total_frames, dtype=int)
        shuffled_indexes = np.random.permutation(indexes)[:max_frames]

        for idx in tqdm(shuffled_indexes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            valid_frame, frame = cap.read()
            if valid_frame:
                img = prepare_image(frame)
                score = self.session.run(None, {self.input_name: img})[0][0][1]
                if score >= threshold:
                    cap.release()
                    return False
        cap.release()
        return True

    def detect_image_paths(self, image_paths, threshold=0.9, max_frames=100):
        total_frames = len(image_paths)
        max_frames = min(total_frames, max_frames)
        indexes = np.arange(total_frames, dtype=int)
        shuffled_indexes = np.random.permutation(indexes)[:max_frames]

        for idx in tqdm(shuffled_indexes):
            frame = cv2.imread(image_paths[idx])
            img = prepare_image(frame)
            score = self.session.run(None, {self.input_name: img})[0][0][1]
            if score >= threshold:
                return False
        return True

    def detect(self, data, data_type, gui=None):
        if data_type == "Image":
            output = self.detect_image(data)
        elif data_type == "Video":
            output = self.detect_video(data)
        elif data_type == "Directory":
            output = self.detect_image_paths(data)
        if not output and gui is not None:
            value = "4e53465720436f6e74656e7420446574656374656421"
            vachakam = bytes.fromhex(value).decode("utf-8")
            sys.stdout.write(vachakam + "\n")
            gui.Warning(vachakam)
        return output
