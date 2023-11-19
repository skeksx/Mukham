import cv2
import numpy as np

ffhq_template = np.array(
    [
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ]
  ],
dtype=np.float32,
)

arcface_template = np.array(
    [
        [
            [0.34191608, 0.46157410],
            [0.65653390, 0.45983392],
            [0.50022500, 0.64050540],
            [0.37097590, 0.82469195],
            [0.63151693, 0.82325090],
        ]
    ],
    dtype=np.float32,
)

mtcnn_template = np.array(
    [
        [
            [0.36562866, 0.46733800],
            [0.63305384, 0.46585885],
            [0.50019130, 0.61942960],
            [0.39032950, 0.77598816],
            [0.61178940, 0.77476320],
        ]
    ],
    dtype=np.float32,
)

set1_template = np.array(
    [
        [
            [0.36718750, 0.45312500],
            [0.64062500, 0.44140625],
            [0.44531250, 0.65234375],
            [0.41015625, 0.78515625],
            [0.62890625, 0.78515625],
        ]
    ],
    dtype=np.float32,
)

set2_template = np.array(
    [
        [
            [0.46108928, 0.44745538],
            [0.51443750, 0.44633930],
            [0.31910715, 0.61613400],
            [0.45675895, 0.79508930],
            [0.50915180, 0.80091080],
        ],
        [
            [0.40206248, 0.44748214],
            [0.58542860, 0.45421430],
            [0.35425892, 0.60813390],
            [0.40336606, 0.76955360],
            [0.57362500, 0.77462500],
        ],
        [
            [0.35473213, 0.45658928],
            [0.64526784, 0.45658928],
            [0.50000000, 0.61154460],
            [0.37913394, 0.77687500],
            [0.62086610, 0.77687500],
        ],
        [
            [0.41825894, 0.45421430],
            [0.60162500, 0.44748214],
            [0.64943750, 0.60813390],
            [0.43006250, 0.77462500],
            [0.60032140, 0.76955360],
        ],
        [
            [0.4892500, 0.44633930],
            [0.5425982, 0.44745538],
            [0.6845803, 0.61613400],
            [0.4945357, 0.80091080],
            [0.5469375, 0.79508930],
        ],
    ]
)


set3_template = np.array(
    [
        [
            [0.27048750, 0.46157410],
            [0.58510536, 0.45983392],
            [0.42879644, 0.64050540],
            [0.29954734, 0.82469195],
            [0.56008840, 0.82325090],
        ]
    ],
    dtype=np.float32,
)


template_map = {
    "arcface": arcface_template,
    "mtcnn": mtcnn_template,
    "set1": set1_template,
    "set2": set2_template,
    "set3": set3_template,
    "ffhq": ffhq_template,
}


def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T


def get_matrix(lmk, templates):
    if templates.shape[0] == 1:
        return umeyama(lmk, templates[0], True)[0:2, :]
    test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_error, best_matrix = float("inf"), []
    for i in np.arange(templates.shape[0]):
        matrix = umeyama(lmk, templates[i], True)[0:2, :]
        error = np.sum(
            np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1))
        )
        if error < min_error:
            min_error, best_matrix = error, matrix
    return best_matrix


def align_crop_with_template(img, lmk, image_size, mode="arcface", scale_factor=1.0):
    templates = template_map[mode] * image_size
    if mode != "ffhq" and image_size % 128 == 0:
        templates *= 0.875
    matrix = get_matrix(lmk, templates)
    warped = cv2.warpAffine(
        img,
        matrix,
        (image_size, image_size),
        borderValue=0.0,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix


def align_crop_generic(img, landmark, image_size, scale_factor=1.0):
    landmark = np.array(landmark)
    eye_left, eye_right = landmark[0], landmark[1]
    eye_avg = (eye_left + eye_right) * 0.5
    mouth_avg = (landmark[3] + landmark[4]) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(
        np.hypot(*eye_to_eye) * 2.0 * scale_factor,
        np.hypot(*eye_to_mouth) * 1.8 * scale_factor,
    )
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    quad_ori = np.copy(quad)
    shrink = int(np.floor(qsize / image_size * 0.5))
    if shrink > 1:
        h, w = img.shape[0:2]
        rsize = (int(np.rint(float(w) / shrink)), int(np.rint(float(h) / shrink)))
        img = cv2.resize(img, rsize, interpolation=cv2.INTER_AREA)
        quad /= shrink
        qsize /= shrink

    h, w = img.shape[0:2]
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        max(int(np.floor(min(quad[:, 0]))) - border, 0),
        max(int(np.floor(min(quad[:, 1]))) - border, 0),
        min(int(np.ceil(max(quad[:, 0]))) + border, w),
        min(int(np.ceil(max(quad[:, 1]))) + border, h),
    )
    if crop[2] - crop[0] < w or crop[3] - crop[1] < h:
        img = img[crop[1] : crop[3], crop[0] : crop[2], :]
        quad -= crop[0:2]

    dst_h, dst_w = image_size, image_size
    template = np.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
    affine_matrix = cv2.estimateAffinePartial2D(quad, template, method=cv2.LMEDS)[0]
    cropped_face = cv2.warpAffine(
        img, affine_matrix, (dst_w, dst_h), borderMode=cv2.BORDER_REPLICATE
    )  # gray
    affine_matrix = cv2.estimateAffinePartial2D(quad_ori, template, method=cv2.LMEDS)[0]

    return cropped_face, affine_matrix


def get_cropped_head(img, landmark, scale=1.25):
    return align_crop_generic(img, landmark, 256, scale_factor=scale)[0]


def align_crop(img, lmk, image_size, method="generic", scale=1.0):
    if method == "generic":
        if image_size % 112 == 0:
            scale *= 0.875
        return align_crop_generic(img, lmk, image_size, scale_factor=scale)
    else:
        lmk = scale_landmark(lmk, scale)
        return align_crop_with_template(
            img, lmk, image_size, mode=method, scale_factor=1.0
        )


def scale_landmark(landmark, scale):
    if scale == 1.0:
        return landmark
    # center = np.mean(landmark, axis=0)
    center = landmark[2]  # nose
    landmark = center + (landmark - center) * scale
    return landmark
