import os
import onnx
import onnxruntime

execution_provider_map = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
    "CoreMLExecutionProvider": "coreml"
    # add more...
}

execution_provider_map_rev = {v: k for k, v in execution_provider_map.items()}

available_onnx_providers = onnxruntime.get_available_providers()
available_onnx_devices = list(set([execution_provider_map.get(p, "cpu") for p in available_onnx_providers]))

def get_onnx_provider(device='cpu', cpu_fallback=True):
    if device not in available_onnx_devices:
        raise ValueError("Unsupported Device")
    provider = [execution_provider_map_rev.get(device, "CPUExecutionProvider")]
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    options.log_verbosity_level = -1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # options.intra_op_num_threads = 1
    # options.inter_op_num_threads = 1
    # options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    if "CUDAExecutionProvider" in provider:
        provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
    if "CPUExecutionProvider" not in provider and cpu_fallback:
        provider.append("CPUExecutionProvider")
    return {"provider":provider, "session_options":options}


class OnnxInferenceSession:
    def __init__(self, model_file=None, session_options=None, provider=["CPUExecutionProvider"]):
        self.model_file = model_file
        self.providers = provider
        self.task_name = ""

        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()

        self.create_session()

    def create_session(self):
        if not hasattr(self, 'session'):
            self.session = onnxruntime.InferenceSession(self.model_file, sess_options=self.session_options, providers=self.providers)
            self.show_primary_provider()

    def delete_session(self):
        if hasattr(self, 'session'):
            del self.session
            self.session = None

    def change_provider(self, providers):
        self.providers = providers
        self.session.set_providers(providers)
        self.show_primary_provider()

    def change_device(self, device_name, fallback_to_cpu=True):
        execution_provider = []
        execution_provider.append(execution_provider_map_rev.get(device_name))
        if fallback_to_cpu:
            execution_provider.append("CPUExecutionProvider")
        self.change_provider(execution_provider)

    def show_primary_provider(self):
        file_name = os.path.basename(self.model_file)
        ep = self.session.get_providers()[0]
        device = execution_provider_map.get(ep)
        print(f"[{device}] {self.task_name}({file_name}) loaded.")

    def ensure_session(self, func, *args, **kwargs):
        try:
            self.create_session()
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error! {e}")

