import cv2
import time
import base64
import numpy as np
from PIL import Image
from functools import lru_cache


def laplacian_blending(A, B, m, levels=7):
    assert A.shape == B.shape
    assert B.shape[:2] == m.shape[:2]
    height = m.shape[0]
    width = m.shape[1]
    size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    size = size_list[np.where(size_list > max(height, width))][0]
    GA = np.zeros((size, size, 3), dtype=np.float32)
    GA[:height, :width, :] = A
    GB = np.zeros((size, size, 3), dtype=np.float32)
    GB[:height, :width, :] = B
    GM = np.zeros((size, size, 3), dtype=np.float32)
    if len(m.shape) == 2:
        GM[:height, :width, 0] = m
        GM[:height, :width, 1] = m
        GM[:height, :width, 2] = m
    else:
        GM[:height, :width, :] = m
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    lpA = [gpA[levels - 1]]
    lpB = [gpB[levels - 1]]
    gpMr = [gpM[levels - 1]]
    for i in range(levels - 1, 0, -1):
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    ls_ = LS[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    ls_ = ls_[:height, :width, :]
    # ls_ = (ls_ - np.min(ls_)) * (255.0 / (np.max(ls_) - np.min(ls_)))
    return ls_.clip(0, 255)


def poisson_blending(source_img, target_img, mask):
    source_img = source_img.astype("uint8")
    target_img = target_img.astype("uint8")
    mask = (mask * 255).astype("uint8")
    br = cv2.boundingRect(cv2.split(mask)[0])
    center = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    output = cv2.seamlessClone(source_img, target_img, mask, center, cv2.NORMAL_CLONE)
    return output


def create_image_grid(images, size=128):
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)

    for i, image in enumerate(images):
        row_idx = (i // num_cols) * size
        col_idx = (i % num_cols) * size
        image = cv2.resize(image.copy(), (size, size))
        if image.dtype != np.uint8:
            image = (image.astype("float32") * 255).astype("uint8")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grid[row_idx : row_idx + size, col_idx : col_idx + size] = image

    return grid


def map_range(value, inMin, inMax, outMin, outMax):
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


def paste_back(
    foreground,
    background,
    matrix,
    custom_mask=None,
    border_crop=(0,0,0,0),
    border_fade_amount=0.2,
    blend_method="alpha blend",
    interpolation=cv2.INTER_LINEAR,
):
    foreground = foreground.clip(0, 255)
    background = background.clip(0, 255)
    inverse_matrix = cv2.invertAffineTransform(matrix)

    f_height, f_width = foreground.shape[:2]
    b_height, b_width = background.shape[:2]

    mask = faded_rectangle(f_width, f_height, border_fade_amount, border_crop)

    if custom_mask is not None:
        custom_mask = cv2.resize(custom_mask, (f_width, f_height)).astype("float32").clip(0, 1)
        mask = np.minimum.reduce([custom_mask, mask])

    foreground = cv2.warpAffine(foreground, inverse_matrix, (b_width, b_height), borderValue=0.0, borderMode=cv2.BORDER_REPLICATE, flags=interpolation)
    mask = cv2.warpAffine(mask, inverse_matrix, (b_width, b_height), borderValue=0.0)
    mask = mask.clip(0, 1)

    if blend_method == "alpha blend":
        composite_image = background
        composite_image[:, :, 0] = mask * foreground[:, :, 0] + (1 - mask) * background[:, :, 0]
        composite_image[:, :, 1] = mask * foreground[:, :, 1] + (1 - mask) * background[:, :, 1]
        composite_image[:, :, 2] = mask * foreground[:, :, 2] + (1 - mask) * background[:, :, 2]
    elif blend_method == "laplacian":
        composite_image = laplacian_blending(foreground, background, mask, levels=10)
    elif blend_method == "poisson":
        composite_image = poisson_blending(foreground, background, mask)
    else:
        raise ValueError("Invalid blend method")

    foreground = foreground.clip(0, 255).astype("uint8")
    composite_image = composite_image.clip(0, 255).astype("uint8")
    return composite_image, foreground, mask


def image_mask_overlay(img, mask):
    img = img.astype("float32") / 255.0
    img *= (mask + 0.25).clip(0, 1)
    img = np.clip(img * 255.0, 0.0, 255.0).astype("uint8")
    return img


def resize_with_padding(img, expected_size=(640, 360), color=(0, 0, 0), max_flip=False):
    original_height, original_width = img.shape[:2]

    if max_flip and original_height > original_width:
        expected_size = (expected_size[1], expected_size[0])

    aspect_ratio = original_width / original_height
    new_width = expected_size[0]
    new_height = int(new_width / aspect_ratio)

    if new_height > expected_size[1]:
        new_height = expected_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(
        resized_img,
        top=(expected_size[1] - new_height) // 2,
        bottom=(expected_size[1] - new_height + 1) // 2,
        left=(expected_size[0] - new_width) // 2,
        right=(expected_size[0] - new_width + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    return canvas


def create_image_grid(images, size=128):
    if images is None or len(images) == 0:
        return None

    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)

    for i, image in enumerate(images):
        row_idx = (i // num_cols) * size
        col_idx = (i % num_cols) * size
        h, w = image.shape[:2]
        if h != w:
            image = center_fit_image(image, (size, size))
        else:
            image = cv2.resize(image, (size, size))
        if image.dtype != np.uint8:
            image = (image.astype("float32") * 255).astype("uint8")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grid[row_idx : row_idx + size, col_idx : col_idx + size] = image

    return grid


def image_to_html(img, size=(640, 360), extension="jpg"):
    if img is not None:
        img = resize_with_padding(img, expected_size=size)
        buffer = cv2.imencode(f".{extension}", img)[1]
        base64_data = base64.b64encode(buffer.tobytes())
        imgbs64 = f"data:image/{extension};base64," + base64_data.decode("utf-8")
        html = '<div style="display: flex; justify-content: center; align-items: center; width: 100%;">'
        html += f'<img src={imgbs64} alt="No Preview" style="max-width: 100%; max-height: 100%;">'
        html += "</div>"
        return html
    return None


def mix_two_image(a, b, opacity=1.0):
    a_dtype = a.dtype
    b_dtype = b.dtype
    a = a.astype("float32")
    b = b.astype("float32")
    if a.shape[:2] != b.shape[:2]:
        a = cv2.resize(a, (b.shape[0], b.shape[1]))
    opacity = min(max(opacity, 0.0), 1.0)
    mixed_img = opacity * b + (1 - opacity) * a
    return mixed_img.astype(a_dtype)


resolution_map = {
    "Original": None,
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}


def resize_image_by_resolution(img, quality):
    resolution = resolution_map.get(quality, None)
    if resolution is None:
        return img

    h, w = img.shape[:2]
    if h > w:
        ratio = resolution[0] / h
    else:
        ratio = resolution[0] / w

    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def fast_pil_encode(pil_image):
    image_arr = np.asarray(pil_image)[:, :, ::-1]
    buffer = cv2.imencode(".jpg", image_arr)[1]
    base64_data = base64.b64encode(buffer.tobytes())
    return "data:image/jpg;base64," + base64_data.decode("utf-8")


def fast_numpy_encode(img_array):
    buffer = cv2.imencode(".jpg", img_array)[1]
    base64_data = base64.b64encode(buffer.tobytes())
    return "data:image/jpg;base64," + base64_data.decode("utf-8")


crf_quality_by_resolution = {
    240: {"poor": 45, "low": 35, "medium": 28, "high": 23, "best": 20},
    360: {"poor": 35, "low": 28, "medium": 23, "high": 20, "best": 18},
    480: {"poor": 28, "low": 23, "medium": 20, "high": 18, "best": 16},
    720: {"poor": 23, "low": 20, "medium": 18, "high": 16, "best": 14},
    1080: {"poor": 20, "low": 18, "medium": 16, "high": 14, "best": 12},
    1440: {"poor": 18, "low": 16, "medium": 14, "high": 12, "best": 10},
    2160: {"poor": 16, "low": 14, "medium": 12, "high": 10, "best": 8},
}


def get_crf_for_resolution(resolution, quality):
    available_resolutions = list(crf_quality_by_resolution.keys())
    closest_resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
    return crf_quality_by_resolution[closest_resolution][quality]


def color_match(target, source, weights=None):
    # TODO
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32") / 255
    trg_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32") / 255

    shape = trg_lab.shape[:2]
    src_lab = cv2.resize(src_lab, shape)

    if weights is None:
        weights = np.ones((src_lab.shape[0], src_lab.shape[1]))
    else:
        weights = cv2.resize(weights, (src_lab.shape[0], src_lab.shape[1])).astype("float32")

    weights = np.tile(np.expand_dims(weights, axis=-1), (1, 1, 3))

    src_avg = np.sum(src_lab * weights, axis=(0, 1)) / np.sum(weights, axis=(0, 1))
    src_std = np.clip(np.std(src_lab, axis=(0, 1)), a_min=(0.0001, 0.0001, 0.0001), a_max=None)
    trg_avg = np.sum(trg_lab * weights, axis=(0, 1)) / np.sum(weights, axis=(0, 1))
    trg_std = np.clip(np.std(trg_lab, axis=(0, 1)), a_min=(0.0001, 0.0001, 0.0001), a_max=None)
    transfer = (((src_std / trg_std) * (trg_lab - trg_avg)) + src_avg) * 255
    return cv2.cvtColor(transfer.clip(0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)


def sharpen(img, sigma, power):
    if power == 0:
        return img
    sigma = max(0, sigma)
    power = max(0, power)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharped = cv2.addWeighted(img, 1.0 + power, blurred, -power, 0)
    return sharped.clip(0, 255)


def blur(img, sigma):
    if sigma == 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return blurred


def median_blur(img, radius):
    if radius <= 0:
        return img
    else:
        dtype = img.dtype
        img = img.astype('uint8')
        return cv2.medianBlur(img, 2 * radius + 1).astype(dtype)


def jpeg_compress(img, quality):
    if quality >= 100:
        return img
    quality = min(max(1, quality), 100)
    ret, result = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ret == True:
        return cv2.imdecode(result, flags=cv2.IMREAD_UNCHANGED)
    return img


def copy_blur(img, ref, multiplier=1):
    gray_image = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    kernel_size = int(np.sqrt(laplacian_var * multiplier)) * 2 + 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def downsample(img, scale_factor):
    if scale_factor >= 1:
        return img
    orginal_size = img.shape[:2]
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    return cv2.resize(cv2.resize(img, new_size), orginal_size)


def estimate_rotation(kps):
    l_eye, r_eye, nose, l_mouth, r_mouth = kps
    eye_center_x = (l_eye[0] + r_eye[0]) / 2
    eye_center_y = (l_eye[1] + r_eye[1]) / 2
    horizontal_angle = np.arctan2(nose[1] - eye_center_y, nose[0] - eye_center_x)
    vertical_angle = np.arctan2(
        nose[1] - (l_mouth[1] + r_mouth[1]) / 2, nose[0] - (l_mouth[0] + r_mouth[0]) / 2
    )
    return np.degrees(horizontal_angle) - 90, np.degrees(vertical_angle) + 90


def erode_blur(img, erode, blur):
    erode, blur = int(erode), int(blur)
    H, W = img.shape[:2]

    if erode > 0:
        el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        iterations = max(1, erode // 2)
        img = cv2.erode(img, el, iterations=iterations)
    elif erode < 0:
        el = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        iterations = max(1, -erode // 2)
        img = cv2.dilate(img, el, iterations=iterations)

    if blur > 0:
        sigma = blur * 0.25
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    return img


@lru_cache()
def faded_rectangle(width=512, height=512, fade_amount=0.2, offset_factor=(0,0,0,0)):
    rectangle = np.ones((width, height), dtype='float32')
    fade_amount = int(width * 0.5 * fade_amount)
    border = max(fade_amount // 2, 1)
    y1 = max(border, int(offset_factor[0] * height))
    y2 = -max(border, int(offset_factor[1] * height))
    x1 = max(border, int(offset_factor[2] * width))
    x2 = -max(border, int(offset_factor[3] * width))
    rectangle[:y1, :] = 0
    rectangle[y2:, :] = 0
    rectangle[:, :x1] = 0
    rectangle[:, x2:] = 0
    if fade_amount > 0:
        rectangle = cv2.GaussianBlur(rectangle, (0, 0), fade_amount * 0.25)
    return rectangle


def smooth_mask_edges(mask, radius=7, iterations=15):
    if radius <= 0 or iterations <= 0:
        return mask
    mask = (mask.clip(0, 1) * 255).astype("uint8")
    mask = cv2.pyrUp(mask)
    for i in range(iterations):
        mask = median_blur(mask, radius)
    mask = cv2.pyrDown(mask)
    return (mask / 255).astype("float32")


def pil_resize(img, size):
    cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv)
    img = img.resize(size, Image.BILINEAR)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def convert_to_3_channel(image):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    alpha = None
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        image = image[:, :, :3]
    return image, alpha

def create_diff_mask(source, target, threshold=10, border=2):
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    diff = target - source
    diff = np.abs(diff).mean(axis=2)
    diff[:border, :] = 0
    diff[-border:, :] = 0
    diff[:, :border] = 0
    diff[:, -border:] = 0
    diff[diff < threshold] = 0
    diff[diff >= threshold] = 255
    diff /= 255
    kernel = np.ones((2,2),np.uint8)
    diff = cv2.dilate(diff,kernel, iterations = 1)
    kernel_size = (5, 5)
    blur_size = tuple(2*i+1 for i in kernel_size)
    diff = cv2.GaussianBlur(diff, blur_size, 0)
    return diff


def center_fit_image(image, size):
    img_height, img_width = image.shape[:2]
    scale_factor_width = size[0] / img_width
    scale_factor_height = size[1] / img_height
    scale_factor = max(scale_factor_width, scale_factor_height)
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    crop_x = max((resized_image.shape[1] - size[0]) // 2, 0)
    crop_y = max((resized_image.shape[0] - size[1]) // 2, 0)
    cropped_image = resized_image[crop_y:crop_y + size[1], crop_x:crop_x + size[0]]
    return cropped_image

def contain_image(image, size):
    if isinstance(image, str):
        image = cv2.imread(image)
    img_height, img_width = image.shape[:2]
    img_aspect = img_width / img_height
    target_aspect = size[0] / size[1]
    if img_aspect > target_aspect:
        scale_factor = size[0] / img_width
    else:
        scale_factor = size[1] / img_height
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    x_offset = (size[0] - resized_image.shape[1]) // 2
    y_offset = (size[1] - resized_image.shape[0]) // 2
    canvas = np.full((size[1], size[0], image.shape[2]), (0, 0, 0), dtype=image.dtype)
    canvas[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image
    return canvas
