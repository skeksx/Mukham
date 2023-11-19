import os
import cv2
import subprocess
import traceback
import numpy as np
from tqdm import tqdm
from . import misc as misc
from . import io as io_util
from . import image as image_util
from .. import default_paths as dp


def get_video_quality(codec, quality):
    # TODO
    codec = codec.lower().replace(".", "").replace(" ", "")
    if codec.startswith("avc"):
        return int((51 - 1) * (100 - quality) / 100) + 1
    elif "264" in codec:
        return int((51 - 1) * (100 - quality) / 100) + 1
    elif "265" in codec:
        return int(51 * quality / 100)
    elif codec in ["hevc", "hev1", "hvc1"]:
        return int(51 * quality / 100)
    elif "mpeg" in codec:
        return int((31 - 1) * (100 - quality) / 100) + 1
    elif codec.startswith("flv"):
        return int((31 - 1) * (100 - quality) / 100) + 1
    elif codec.startswith("wmv"):
        return int((31 - 1) * (100 - quality) / 100) + 1
    elif codec in ["theora", "vp6", "vp7", "vp8", "vp9"]:
        return int(63 * quality / 100)
    elif codec.startswith("prores"):
        return int((31 - 1) * (100 - quality) / 100) + 1
    elif "xvid" in codec:
        return int((31 - 1) * (100 - quality) / 100) + 1
    elif codec in ["dnxhd", "rv10", "rv30", "rv40"]:
        return int(51 * quality / 100)
    elif "mjpeg" in codec:
        return int(63 * quality / 100)
    else:
        return 3

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    audio = cap.get(cv2.CAP_PROP_AUDIO_STREAM)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_name = chr(codec & 0xFF) + chr((codec >> 8) & 0xFF) + chr((codec >> 16) & 0xFF) + chr((codec >> 24) & 0xFF)
    cap.release()
    return total_frames, fps, resolution, codec_name.lower()

def get_single_video_frame(video_path, frame_index, swap_rgb=False):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(int(frame_index), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    valid_frame, frame = cap.read()
    cap.release()
    if valid_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if swap_rgb:
            frame = frame[:, :, ::-1]
        return frame
    return None

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def create_video_thumbnail(video_file, size=(128,128)):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    thumbanil = None
    if ret:
        thumbanil = image_util.center_fit_image(frame, size)
    cap.release()
    if thumbanil is None:
        return np.zeros((*size, 3), dtype=np.uint8)
    return thumbanil

def compute_resized_resolution(original_resolution, size):
    if isinstance(size, str):
        size = size.replace("p","")
    size = int(size)
    original_width, original_height = original_resolution
    if original_width < original_height:
        new_resolution = [size, int(size * (original_height / original_width))]
    else:
        new_resolution = [int(size * (original_width / original_height)), size]
    if new_resolution[0] % 2 != 0: new_resolution[0] += 1
    if new_resolution[1] % 2 != 0: new_resolution[1] += 1
    return new_resolution

@misc.spinner(text="Extracting Frames")
def ffmpeg_extract_frames(
    video_path,
    destination,
    start_frame=None,
    end_frame=None,
    custom_fps=None,
    custom_resolution=None,
    quality=90,
    name="frame_%d.jpg",
):
    ffmpeg_path = dp.FFMPEG_PATH
    total_frames, fps, original_resolution, codec = get_video_info(video_path)
    start_frame = 0 if start_frame is None else min(start_frame, total_frames - 1)
    end_frame = (total_frames - 1) if end_frame is None else min(end_frame, total_frames - 1)
    if custom_fps is not None:
        fps = str(custom_fps)
    quality = get_video_quality(codec, quality)
    cmd = [ffmpeg_path, "-loglevel", "info", "-hwaccel", "auto"]
    cmd.extend(["-i", video_path])
    cmd.extend(["-q:v", str(quality)])
    if custom_resolution is not None:
        new_resolution = compute_resized_resolution(original_resolution, custom_resolution)
        cmd.extend(["-s", f"{new_resolution[0]}x{new_resolution[1]}"])
    cmd.extend(["-pix_fmt", "rgb24"])
    cmd.extend(["-vf", f'trim=start_frame={start_frame}:end_frame={end_frame},fps={fps}'])
    cmd.extend(['-vsync', '0'])
    cmd.extend(["-y", os.path.join(destination, name)])
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, io_util.get_images_from_directory(destination)
    except subprocess.CalledProcessError as e:
        print("\nError extracting frames!")
        print(e)
    return False, None


@misc.spinner(text="Merging Frames")
def ffmpeg_merge_frames(seq_directory, pattern, destination, fps=30, quality=90, codec="libx264"):
    ffmpeg_path = dp.FFMPEG_PATH
    rate_arg = "-cq" if "nvenc" in codec else "-crf"
    quality = get_video_quality(codec, quality)
    cmd = [
        ffmpeg_path,
        "-loglevel", "info",
        "-hwaccel", "auto",
        "-r", str(fps),
        "-i", os.path.join(seq_directory, pattern),
        "-c:v", codec,
        rate_arg, str(quality),
        "-pix_fmt", "yuv420p",
        "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
        "-y", destination,
    ]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, destination
    except Exception as e:
        print(f"\nError merging frames!")
        print(e)
    return False, None

@misc.spinner(text="Merging Audio")
def ffmpeg_mux_audio(source, target, destination, start_frame=None, end_frame=None):
    ffmpeg_path = dp.FFMPEG_PATH
    total_frames, fps, resolution, codec = get_video_info(source)
    if start_frame is None:
        start_frame = 0
    else:
        start_frame = min(start_frame, total_frames - 1)
    if end_frame is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(end_frame, total_frames - 1)
    extracted_audio_path = os.path.join(os.path.dirname(destination), "extracted_audio.aac")
    cmd1 = [
        ffmpeg_path,
        "-loglevel", "info",
        "-i", source,
        "-ss", str(start_frame / fps),
        "-to", str(end_frame / fps),
        "-vn",
        "-c:a", "aac",
        "-y", extracted_audio_path,
    ]
    try:
        subprocess.check_output(cmd1, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"\nError merging audio!")
        return False, None
    cmd2 = [
        ffmpeg_path,
        "-loglevel", "info",
        "-i", target,
        "-i", extracted_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        "-shortest",
        destination
    ]
    try:
        subprocess.check_output(cmd2, stderr=subprocess.STDOUT)
        output = (True, destination)
    except subprocess.CalledProcessError as e:
        print(f"\nError merging audio!")
        output = (False, None)
    io_util.remove_file(extracted_audio_path)
    return output

@misc.spinner(text="Converting to GIF")
def ffmpeg_replace_to_gif(video_path):
    ffmpeg_path = dp.FFMPEG_PATH
    base_name, video_extension = os.path.splitext(video_path)
    gif_path = base_name + ".gif"
    cmd = [
        ffmpeg_path,
        "-loglevel", "info",
        "-i", video_path,
        "-y", gif_path
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        output = (True, gif_path)
    except subprocess.CalledProcessError as e:
        print(f"\nError creating gif!")
        print(e)
        output = (False, None)
    io_util.remove_file(video_path)
    return output

@misc.spinner(text="Merging Frames to GIF")
def ffmpeg_merge_frames_to_gif(seq_directory, pattern, destination, fps=30):
    ffmpeg_path = dp.FFMPEG_PATH
    # dither_types = ["none", "bayer", "sierra2_4a", "floyd_steinberg", "atkinson"]
    cmd = [
        ffmpeg_path,
        '-r', str(fps),
        '-i', os.path.join(seq_directory, pattern),
        # '-vf', f'fps={str(fps)},split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither={dither_types[1]}',
        '-vf', f'fps={str(fps)},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-loop', '0',
        '-y', destination
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True, destination
    except Exception as e:
        print(f"\nError merging frames!")
        print(e)
        return False, None

class FFMpegVideoWriter:
    def __init__(self, output_file, fps, resolution, codec="libx264"):
        ffmpeg_path = dp.FFMPEG_PATH
        self.ffin = subprocess.Popen(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                "{}x{}".format(resolution[0], resolution[1]),
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "pipe:",
                "-an",
                "-vcodec",
                codec,
                output_file,
            ],
            stdin=subprocess.PIPE,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def release(self):
        self.ffin.stdin.close()
        self.ffin.wait()

    def write(self, frame, swap_rgb=True):
        if swap_rgb:
            frame = frame[:, :, ::-1]
        self.ffin.stdin.write(frame.tobytes())