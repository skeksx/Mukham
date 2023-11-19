import os
import cv2
import time
import glob
import shutil
import platform
import requests
import webbrowser
import subprocess
from tqdm import tqdm
from datetime import datetime


image_extensions = [
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".tiff",
    ".tif",
    ".webp",
    ".pic",
    ".ico",
]

video_extensions = [
    ".avi",
    ".mkv",
    ".mp4",
    ".mov",
    ".wmv",
    ".flv",
    ".3gp",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".asf",
    ".ts",
    ".vob",
    ".webm",
    ".gif",
]


def get_file_extension(file_path):
    assert "." in file_path, f"Error finding file extension: {file_path}"
    return os.path.splitext(file_path)[1].replace(" ", "").lower()

def filter_file_by_extension(files):
    dfms, safetensors, images, videos = [], [], [], []
    for f in files:
        ext = get_file_extension(f)
        if ext == ".dfm":
            dfms.append(f)
        elif ext == ".safetensors":
            safetensors.append(f)
        elif ext in image_extensions:
            images.append(f)
        elif ext in video_extensions:
            videos.append(f)
        else:
            assert False, f"Unsupported file extension: {ext}"
    return dfms, safetensors, images, videos

def get_images_from_directory(directory_path):
    file_paths = []
    for file_path in glob.glob(os.path.join(directory_path, "*")):
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths

def get_videos_from_directory(directory_path):
    file_paths = []
    for file_path in glob.glob(os.path.join(directory_path, "*")):
        if any(file_path.lower().endswith(ext) for ext in video_extensions):
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths

def get_files_by_extension(directory_path, extension):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        files = os.listdir(directory_path)
        file_names = [f for f in files if f.lower().endswith(extension.lower())]
        if file_names:
            full_paths = [
                os.path.join(directory_path, file_name) for file_name in file_names
            ]
            return full_paths, file_names
    return [], []

def copy_files_to_directory(files, destination):
    file_paths = []
    for file_path in files:
        new_file_path = shutil.copy(file_path, destination)
        file_paths.append(new_file_path)
    return file_paths

def create_directory(directory_path, remove_existing=True):
    if os.path.exists(directory_path) and remove_existing:
        shutil.rmtree(directory_path)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        return directory_path
    else:
        counter = 1
        while True:
            new_directory_path = f"{directory_path}_{counter}"
            if not os.path.exists(new_directory_path):
                os.mkdir(new_directory_path)
                return new_directory_path
            counter += 1

def add_datetime_to_filename(filename):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name, file_extension = os.path.splitext(filename)
    new_filename = f"{file_name}_{formatted_datetime}{file_extension}"
    return new_filename

def open_file(file_path):
    if not os.path.exists(file_path):
        return
    webbrowser.open(file_path)

def open_directory(directory_path):
    system = platform.system().lower()
    if system == "linux":
        subprocess.Popen(['xdg-open', directory_path])
    elif system == "darwin":
        subprocess.Popen(['open', directory_path])
    elif system == "windows":
        subprocess.Popen(['explorer', directory_path], shell=True)
    else:
        print("Unknown OS")

def browse_file(*args, **kwargs):
    import plyer
    path = plyer.filechooser.open_file(*args, **kwargs)
    if isinstance(path, list):
        return path[0]

def browse_directory(*args, **kwargs):
    import plyer
    path = plyer.filechooser.choose_dir(*args, **kwargs)
    if isinstance(path, list):
        return path[0]

def save_file(*args, **kwargs):
    import plyer
    path = plyer.filechooser.save_file(*args, **kwargs)
    if isinstance(path, list):
        return path[0]

def is_file(file_path):
    if not file_path:
        return False
    return os.path.isfile(file_path)

def is_image(file_path):
    if is_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in image_extensions
    return False

def is_video(file_path):
    if is_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in video_extensions
    return False

def is_safetensor(file_path):
    if is_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in [".safetensors"]
    return False

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)

def remove_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error: {e}")

def is_valid_directory(path):
    return os.path.exists(path) and os.path.isdir(path)


def download_file(url, destination):
    directory_name = os.path.dirname(destination)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError("Failed downloading url %s" % url)

    total_length = response.headers.get('content-length')
    with open(destination, 'wb') as f:
        if total_length is None:
            print(f"Downloading {url}")
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
            print("Downloading completed")
        else:
            total_length = int(total_length)
            with tqdm(total=total_length, unit='B', unit_scale=True, dynamic_ncols=True) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
