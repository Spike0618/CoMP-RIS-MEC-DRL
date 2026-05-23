"""



-  numpy
-  scipy Savitzky-Golay  Gaussian 
-  scipy EMA//
"""

from typing import Tuple, Optional
import numpy as np

try:
    from scipy.signal import savgol_filter  # type: ignore
except Exception:
    savgol_filter = None

try:
    from scipy.ndimage import gaussian_filter1d  # type: ignore
except Exception:
    gaussian_filter1d = None


def exponential_moving_average(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    EMA

    
    - data
    - alpha (0, 1] 0.02~0.1

    
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return data
    
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def savitzky_golay_smooth(data: np.ndarray, window_length: Optional[int] = None, 
                          polyorder: int = 3) -> np.ndarray:
    """
    Savitzky-Golay 

    
    - data
    - window_lengthNone 
    - polyorder 2~4

     scipy 
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size < 5:
        return data
    
    if window_length is None:
        
        window_length = max(5, min(51, int(len(data) * 0.07)))
        if window_length % 2 == 0:
            window_length += 1
    
    
    window_length = min(window_length, len(data))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(5, window_length)
    
    
    polyorder = min(polyorder, window_length - 1)
    
    if savgol_filter is None:
        
        return moving_average_smooth(data, window_length)

    try:
        return savgol_filter(data, window_length, polyorder, mode='interp')  # type: ignore[misc]
    except Exception:
        
        return moving_average_smooth(data, window_length)


def gaussian_smooth(data: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    

    
    - data
    - sigmaNone 

     scipy 
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size < 3:
        return data
    
    if sigma is None:
        
        sigma = max(1.0, len(data) * 0.03)
    
    if gaussian_filter1d is None:
        
        if sigma is None:
            alpha = 0.05
        else:
            alpha = float(np.clip(1.0 / (float(sigma) + 1.0), 0.02, 0.20))
        return exponential_moving_average(data, alpha=alpha)

    return gaussian_filter1d(data, sigma=sigma, mode='nearest')  # type: ignore[misc]


def moving_average_smooth(data: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    

    
    - data
    - windowNone 

     len(data) - window + 1
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return data
    
    if window is None:
        window = max(1, min(50, int(len(data) * 0.05)))
    
    window = max(1, min(window, len(data)))
    
    if window == 1:
        return data
    
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode='valid')


def compute_confidence_band(data: np.ndarray, window: int = 50,
                            confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    CI

    
    - 
    - CI
    - CIseed/episode

    
    - data
    - window
    - confidence 0.95  1.96 

     (lower, upper)
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    n = len(data)
    
    if n < window:
        
        std = np.std(data) if n > 1 else 0.0
        z = 1.96 if confidence >= 0.95 else 1.645
        margin = z * std
        return data - margin, data + margin
    
    lower = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_data = data[start:end]
        
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        
        z = 1.96 if confidence >= 0.95 else 1.645
        margin = z * std / np.sqrt(len(window_data))
        
        lower[i] = mean - margin
        upper[i] = mean + margin
    
    return lower, upper


def smooth_with_confidence(data: np.ndarray, method: str = 'ema', 
                          alpha: float = 0.05, window: Optional[int] = None,
                          compute_ci: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    

    
    - data
    - method'ema' / 'savgol' / 'gaussian' / 'ma'
    - alphaEMA  method='ema' 
    - window
    - compute_ci

     (smoothed, lower_ci, upper_ci)
    -  compute_ci=False lower_ci/upper_ci  None
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    
    if data.size == 0:
        return data, None, None
    
    
    if method == 'ema':
        smoothed = exponential_moving_average(data, alpha=alpha)
    elif method == 'savgol':
        smoothed = savitzky_golay_smooth(data, window_length=window)
    elif method == 'gaussian':
        sigma = window if window is not None else None
        smoothed = gaussian_smooth(data, sigma=sigma)
    elif method == 'ma':
        smoothed = moving_average_smooth(data, window=window)
        
        if len(smoothed) < len(data):
            pad_len = len(data) - len(smoothed)
            smoothed = np.concatenate([data[:pad_len], smoothed])
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    
    if compute_ci and len(data) > 10:
        ci_window = window if window is not None else max(10, int(len(data) * 0.05))
        lower, upper = compute_confidence_band(data, window=ci_window)
    else:
        lower, upper = None, None
    
    return smoothed, lower, upper


def filter_valid_xy(
    x: np.ndarray,
    y: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    invalid_values: Tuple[float, ...] = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    /PhaseF 

    
    1)  valid_mask mask 
    2)  NaN/Infx  y 
    3)  invalid_values 0.0

    
    - x_f, y_f
    - keep_mask (x,y)  CI/STD 
    """
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(x0.size, y0.size))
    x0 = x0[:n]
    y0 = y0[:n]

    keep = np.ones((n,), dtype=bool)
    if valid_mask is not None:
        vm = np.asarray(valid_mask).reshape(-1).astype(bool)
        if vm.size != n:
            vm = np.resize(vm, (n,))
        keep &= vm

    keep &= np.isfinite(x0) & np.isfinite(y0)

    if invalid_values:
        bad = np.zeros((n,), dtype=bool)
        for v in invalid_values:
            try:
                bad |= (y0 == float(v))
            except Exception:
                continue
        keep &= ~bad

    return x0[keep], y0[keep], keep


def gaussian_smooth_with_band(
    data: np.ndarray,
    sigma: float = 250.0,
    rolling_window: int = 800,
    band_scale: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     + 

     EMA
     seed 

    
    - data Average Episode Reward
    - sigma15000  200~300
    - rolling_window
    - band_scale0.3  0.3_local

    (smoothed_mean, lower_band, upper_band)
    """
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    n = data.size
    if n == 0:
        return data, data, data

    if gaussian_filter1d is None:
        
        sm = exponential_moving_average(data, alpha=max(0.01, 2.0 / (sigma + 1.0)))
        return sm, sm, sm

    
    smoothed = gaussian_filter1d(data, sigma=sigma, mode='nearest')

    
    half_w = rolling_window // 2
    rolling_std = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        rolling_std[i] = np.std(data[lo:hi])

    
    std_smooth = gaussian_filter1d(rolling_std, sigma=sigma, mode='nearest')

    upper = smoothed + band_scale * std_smooth
    lower = smoothed - band_scale * std_smooth

    return smoothed, lower, upper


def ema_smooth(y: np.ndarray, span: int = 8, adjust: bool = False) -> np.ndarray:
    """
     span  EMA 

    
    - span  PhaseF  eval  span=8
    - adjust=False 
    """
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    if y0.size == 0:
        return y0

    span = int(max(1, span))
    if span <= 1 or y0.size <= 1:
        return y0.copy()

    alpha = float(2.0 / (float(span) + 1.0))
    out = np.empty_like(y0, dtype=np.float64)

    if not adjust:
        out[0] = float(y0[0])
        for i in range(1, y0.size):
            yi = float(y0[i])
            if not np.isfinite(yi):
                out[i] = out[i - 1]
            else:
                out[i] = alpha * yi + (1.0 - alpha) * float(out[i - 1])
        return out

    
    num = 0.0
    den = 0.0
    for i in range(y0.size):
        yi = float(y0[i])
        if not np.isfinite(yi):
            out[i] = out[i - 1] if i > 0 else float("nan")
            continue
        num = alpha * yi + (1.0 - alpha) * num
        den = alpha + (1.0 - alpha) * den
        out[i] = num / max(den, 1e-12)
    return out


def adaptive_ema_span(span_cfg: int, n_points: int) -> Optional[int]:
    """
    EMAspan

    
    - span_eff = min(span_cfg, max(3, n_points//2))
    -  n_points < 3 NoneEMAtoo few points
    """
    n_points = int(n_points)
    if n_points < 3:
        return None
    span_cfg = int(max(1, span_cfg))
    span_eff = int(min(span_cfg, max(3, n_points // 2)))
    return int(max(1, span_eff))


def ema_smooth_adaptive(
    y: np.ndarray,
    span_cfg: int = 8,
    adjust: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    EMA(ema, span_eff)

    - <3 (None, None)too few points
    """
    y0 = np.asarray(y, dtype=np.float64).reshape(-1)
    span_eff = adaptive_ema_span(int(span_cfg), int(y0.size))
    if span_eff is None:
        return None, None
    return ema_smooth(y0, span=int(span_eff), adjust=adjust), int(span_eff)
