import time
import onnx
import cv2
import onnxruntime
import numpy as np
from onnx import numpy_helper
from swap_mukham.utils.device import OnnxInferenceSession


class Inswapper(OnnxInferenceSession):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.crop_size = 128
        self.align_template = "arcface"

    def forward(self, target, source):
        trg = cv2.resize(target, (128, 128))

        latent = source["embedding"].reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        blob = trg.astype("float32") / 255
        blob = blob[:, :, ::-1]
        blob = np.expand_dims(blob, axis=0).transpose(0, 3, 1, 2)

        # io_binding = self.session.io_binding()
        # io_binding.bind_cpu_input("target", blob)
        # io_binding.bind_cpu_input("source", latent)
        # io_binding.bind_output("output", "cuda")
        # self.session.run_with_iobinding(io_binding)
        # result = io_binding.copy_outputs_to_cpu()[0]

        result = self.session.run(["output"], {"target": blob, "source": latent})[0]

        out = result[0].transpose((1, 2, 0))
        out = (out * 255).clip(0, 255)

        del blob, latent, trg

        return out[:, :, ::-1]
