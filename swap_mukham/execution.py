import os
import cv2
import time
import queue
import threading
import subprocess
import concurrent.futures
from tqdm import tqdm

import json
import hashlib
from functools import wraps
from collections import OrderedDict

# TODO
class Cache:
    def __init__(self, max_size=None):
        self.cache = {}
        self.max_size = max_size

    def evict_oldest(self):
        if self.max_size is not None and len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]


def cache_function(max_size=None):
    cache = Cache(max_size)
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            key_components = []
            for arg in args:
                if isinstance(arg, OrderedDict):
                    key_components.append(f'OrderedDict-{json.dumps(arg, sort_keys=True)}')
                elif isinstance(arg, np.ndarray):
                    key_components.append(f'NumPyArray-{arg.tobytes()}')
                elif isinstance(arg, (str, int, float)):
                    key_components.append(f'{arg}-{type(arg).__name__}')
                else:
                    key_components.append(repr(arg))
            for key, value in kwargs.items():
                key_components.append(f'{key}-{value}-{type(value).__name__}')
            cache_key = hashlib.sha256('|'.join(key_components).encode()).hexdigest()
            if cache_key in cache.cache:
                print("Result retrieved from cache.")
                return cache.cache[cache_key]
            result = func(self, *args, **kwargs)
            cache.cache[cache_key] = result
            cache.evict_oldest()
            return result
        return wrapper
    return decorator


class OneProducerChainedConsumer:
    def __init__(self, producer_func, chain_funcs, length):
        self.length = length
        self.producer_func = producer_func
        self.chain_funcs = chain_funcs
        self.queues = [queue.Queue() for _ in range(len(self.chain_funcs))]
        # self.locks = [threading.Lock() for _ in range(len(self.chain_funcs))]

    def process(self, num_workers=6):
        start_time = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                executor.submit(self.error_wrapper, self.producer_func, "producer_func", self)

                futures = []
                for _ in range(self.length):
                    for func_idx, func in enumerate(self.chain_funcs):
                        futures.append(executor.submit(self.error_wrapper, func, func.__name__, self, func_idx,))

                concurrent.futures.wait(futures)
        except Exception as e:
            print(f"Error in OneProducerChainedConsumer: {e}")
        print(time.time() - start_time)

    @staticmethod
    def error_wrapper(func, func_name, *args):
        try:
            func(*args)
        except Exception as e:
            print(f"Error in {func_name}: {e}")


class ThreadedExecutor:
    def __init__(self, processor, data_list):
        self.processor = processor
        self.total = len(data_list)
        self.queue = queue.Queue()

        for data in data_list:
            self.queue.put(data)

    def run(self, threads=4, text="processing"):
        self.batch_size = int(max(self.total // threads, 1))
        with tqdm(total = self.total, desc=text, unit='frame', dynamic_ncols=True) as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers = threads) as executor:
                futures = []
                while not self.queue.empty():
                    future = executor.submit(self.__process, self.batch, progress)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    def __process(self, data, progress):
        for d in data:
            self.processor(d)
            progress.update(1)

    @property
    def batch(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                batch.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return batch



class ThreadedVideoProcessor:
    def __init__(self, processor, data, mode='video'):
        self.processor = processor
        self.data = data
        self.frame_queue = queue.Queue()

        if mode == 'video':
            cap = cv2.VideoCapture(data)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.processing_func = self.process_video
            for frame_idx in range(0, self.total_frames):
                self.frame_queue.put(frame_idx)
        elif mode == 'list':
            self.total_frames = len(data)
            self.processing_func = self.process_list
            for i, data in enumerate(self.data):
                self.frame_queue.put((data, i))

    def run(self, threads=4):
        self.batch_size = int(max(self.total_frames // threads, 1))
        with tqdm(total = self.total_frames, desc='processing', unit='frame', dynamic_ncols=True) as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers = threads) as executor:
                futures = []
                while not self.frame_queue.empty():
                    batch = self.get_batch()
                    future = executor.submit(self.processing_func, batch, progress)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    def get_batch(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                data = self.frame_queue.get_nowait()
                batch.append(data)
        except queue.Empty:
            pass
        return batch

    def process_video(self, indices, progress):
        cap = cv2.VideoCapture(self.data)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = cap.read()
            self.processor(frame, index)
            progress.update(1)
        cap.release()

    def process_list(self, data, progress):
        for frame, frame_index in data:
            self.processor(frame, frame_index)
            progress.update(1)
