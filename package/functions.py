import cv2
import numpy as np
from PIL import Image, ImageTk


def load_image(file_path: str) -> np.ndarray | None:
    try:
        img_array = np.fromfile(file_path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cv_to_tk(img: np.ndarray, max_size: tuple[int, int] = (460, 400)) -> ImageTk.PhotoImage:
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def gray_basic(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()


def gray_invert(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    return 255 - gray


def gray_binary(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def color_red(img: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    bgr[:, :, 0] = 0
    bgr[:, :, 1] = 0
    return bgr


def color_green(img: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    bgr[:, :, 0] = 0
    bgr[:, :, 2] = 0
    return bgr


def color_blue(img: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    bgr[:, :, 1] = 0
    bgr[:, :, 2] = 0
    return bgr


def color_original(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()


def blur_average(img: np.ndarray) -> np.ndarray:
    return cv2.blur(img, (7, 7))


def blur_gaussian(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (9, 9), 0)


def blur_box(img: np.ndarray) -> np.ndarray:
    return cv2.boxFilter(img, -1, (9, 9))


def sharpen_unsharp(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def sharpen_laplacian(img: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(img, cv2.CV_64F)
    res = img.astype(np.float64) - 0.5 * lap
    return np.clip(res, 0, 255).astype(np.uint8)


def sharpen_strong(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
    return cv2.filter2D(img, -1, kernel)


def edge_sobel(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.clip(cv2.magnitude(sx, sy), 0, 255).astype(np.uint8)


def edge_laplacian(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(lap)


def edge_canny(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    return cv2.Canny(gray, 100, 200)


def filter_min(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel)


def filter_max(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel)


def filter_median(img: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(img, 5)


def filter_freq_lpf(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    mask = np.zeros((rows, cols, 2), np.float32)
    cv2.circle(mask, (cols // 2, rows // 2), 25, (1, 1), -1)
    idft = cv2.idft(np.fft.ifftshift(dft_shift * mask))
    mag = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    return cv2.convertScaleAbs(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))


def filter_freq_hpf(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    mask = np.ones((rows, cols, 2), np.float32)
    cv2.circle(mask, (cols // 2, rows // 2), 25, (0, 0), -1)
    idft = cv2.idft(np.fft.ifftshift(dft_shift * mask))
    mag = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    return cv2.convertScaleAbs(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))


def morph_erode(img: np.ndarray) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.erode(img, ker)


def morph_dilate(img: np.ndarray) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(img, ker)


def morph_open(img: np.ndarray) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, ker)


def morph_close(img: np.ndarray) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker)


def scale_down_half(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2))
    canvas = np.zeros_like(img)
    y_off, x_off = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y_off : y_off + small.shape[0], x_off : x_off + small.shape[1]] = small
    return canvas


def scale_up_1_5(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    big = cv2.resize(img, None, fx=1.5, fy=1.5)
    sy, sx = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return big[sy : sy + h, sx : sx + w]


def scale_up_2_0(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    big = cv2.resize(img, None, fx=2.0, fy=2.0)
    sy, sx = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return big[sy : sy + h, sx : sx + w]


def _interp_compare(img: np.ndarray, flag: int) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=flag)


def interp_nearest(img: np.ndarray) -> np.ndarray:
    return _interp_compare(img, cv2.INTER_NEAREST)


def interp_linear(img: np.ndarray) -> np.ndarray:
    return _interp_compare(img, cv2.INTER_LINEAR)


def interp_cubic(img: np.ndarray) -> np.ndarray:
    return _interp_compare(img, cv2.INTER_CUBIC)


def translate_right(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, 60], [0, 1, 0]]), (w, h))


def translate_down(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, 60]]), (w, h))


def translate_diag(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, 50], [0, 1, 50]]), (w, h))


def rotate_90(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate_180(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate_270(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_45(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def bright_up(img: np.ndarray) -> np.ndarray:
    return cv2.add(img, 50)


def bright_down(img: np.ndarray) -> np.ndarray:
    return cv2.subtract(img, 50)


def contrast_up(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.6, beta=0)


def contrast_down(img: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=0.5, beta=0)


def contrast_hist_eq(img: np.ndarray) -> np.ndarray:
    gray = gray_basic(img)
    return cv2.equalizeHist(gray)


def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    if len(img1.shape) != len(img2_resized.shape):
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img1, alpha, img2_resized, 1.0 - alpha, 0)