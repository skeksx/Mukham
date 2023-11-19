import time
import threading
import numpy as np

def spinner(text="Processing"):
    def display_spinner():
        spin_chars = "/-\\|"
        while not done:
            for char in spin_chars:
                if done:
                    break
                print(f"\r{text} {char}", end="")
                time.sleep(0.1)

    def finish_spinner(success=True):
        global done
        done = True
        finish_text = text if success else "Error"
        print(f"\r{finish_text}")

    def decorator_with_args(function):
        def decorated_function(*args, **kwargs):
            global done
            done = False
            spinner_thread = threading.Thread(target=display_spinner)
            spinner_thread.daemon = True
            spinner_thread.start()
            try:
                result = function(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                print(f"\r{text} Error: {str(e)}")
            finally:
                finish_spinner(success)
            return result
        return decorated_function

    return decorator_with_args


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} execution time: {execution_time:.6f} seconds")
        return result
    return wrapper


data_type_bytes = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'float16': 2, 'float32': 4}

def estimate_max_batch_size(resolution, chunk_size=1024, data_type='float32', channels=3):
    pixel_size = data_type_bytes.get(data_type, 1)
    image_size = resolution[0] * resolution[1] * pixel_size * channels
    number_of_batches = (chunk_size * 1024 * 1024) // image_size
    return max(number_of_batches, 1)