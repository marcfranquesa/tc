from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Code 128 symbol patterns (values 0..105) expressed as 11-module bitstrings.
# Sourced from the (MIT-licensed) python-barcode project:
# https://github.com/WhyNotHugo/python-barcode/blob/master/barcode/charsets/code128.py
#
# We only use this as a static lookup table; decoding logic is implemented from scratch.
CODE128_BITS: Tuple[str, ...] = (
    "11011001100",
    "11001101100",
    "11001100110",
    "10010011000",
    "10010001100",
    "10001001100",
    "10011001000",
    "10011000100",
    "10001100100",
    "11001001000",
    "11001000100",
    "11000100100",
    "10110011100",
    "10011011100",
    "10011001110",
    "10111001100",
    "10011101100",
    "10011100110",
    "11001110010",
    "11001011100",
    "11001001110",
    "11011100100",
    "11001110100",
    "11101101110",
    "11101001100",
    "11100101100",
    "11100100110",
    "11101100100",
    "11100110100",
    "11100110010",
    "11011011000",
    "11011000110",
    "11000110110",
    "10100011000",
    "10001011000",
    "10001000110",
    "10110001000",
    "10001101000",
    "10001100010",
    "11010001000",
    "11000101000",
    "11000100010",
    "10110111000",
    "10110001110",
    "10001101110",
    "10111011000",
    "10111000110",
    "10001110110",
    "11101110110",
    "11010001110",
    "11000101110",
    "11011101000",
    "11011100010",
    "11011101110",
    "11101011000",
    "11101000110",
    "11100010110",
    "11101101000",
    "11101100010",
    "11100011010",
    "11101111010",
    "11001000010",
    "11110001010",
    "10100110000",
    "10100001100",
    "10010110000",
    "10010000110",
    "10000101100",
    "10000100110",
    "10110010000",
    "10110000100",
    "10011010000",
    "10011000010",
    "10000110100",
    "10000110010",
    "11000010010",
    "11001010000",
    "11110111010",
    "11000010100",
    "10001111010",
    "10100111100",
    "10010111100",
    "10010011110",
    "10111100100",
    "10011110100",
    "10011110010",
    "11110100100",
    "11110010100",
    "11110010010",
    "11011011110",
    "11011110110",
    "11110110110",
    "10101111000",
    "10100011110",
    "10001011110",
    "10111101000",
    "10111100010",
    "11110101000",
    "11110100010",
    "10111011110",
    "10111101110",
    "11101011110",
    "11110101110",
    "11010000100",
    "11010010000",
    "11010011100",
)

# Stop pattern is 13 modules (followed by a 2-module termination bar in printing).
CODE128_STOP_BITS: str = "1100011101011"


def _bits_to_runwidths(bits: str) -> Tuple[int, ...]:
    runs: List[int] = []
    cur = bits[0]
    n = 1
    for b in bits[1:]:
        if b == cur:
            n += 1
        else:
            runs.append(n)
            cur = b
            n = 1
    runs.append(n)
    return tuple(runs)


CODE128_RUNS: Tuple[Tuple[int, ...], ...] = tuple(
    _bits_to_runwidths(b) for b in CODE128_BITS
)
CODE128_STOP_RUNS: Tuple[int, ...] = _bits_to_runwidths(CODE128_STOP_BITS)

# Build reverse lookup from run widths to code value.
_RUNS_TO_VALUE = {runs: i for i, runs in enumerate(CODE128_RUNS)}


@dataclass(frozen=True)
class Code128Result:
    text: str
    codes: Tuple[int, ...]  # includes start..checksum, excludes stop
    start_value: int
    checksum_value: int
    direction: str  # "lr" or "rl"


@lru_cache(maxsize=None)
def _valid_module_tuples(
    n_runs: int, modules_total: int
) -> Tuple[Tuple[int, ...], ...]:
    """
    Enumerate all tuples of length n_runs with entries 1..4 that sum to modules_total.
    Cached because we call it a lot.
    """

    out: List[Tuple[int, ...]] = []

    def rec(i: int, remaining: int, cur: List[int]) -> None:
        if i == n_runs:
            if remaining == 0:
                out.append(tuple(cur))
            return
        # prune min/max possible
        min_possible = 1 * (n_runs - i - 1)
        max_possible = 4 * (n_runs - i - 1)
        for v in range(1, 5):
            rem2 = remaining - v
            if rem2 < 0:
                break
            if rem2 < min_possible or rem2 > max_possible:
                continue
            cur.append(v)
            rec(i + 1, rem2, cur)
            cur.pop()

    rec(0, modules_total, [])
    return tuple(out)


def _best_modules_for_runs(
    run_px: Sequence[int], modules_total: int
) -> Optional[Tuple[Tuple[int, ...], float]]:
    """
    Given pixel run lengths, find the module-width tuple (1..4, fixed sum) that best fits.
    Returns (module_tuple, normalized_error). Lower error is better.
    """
    n = len(run_px)
    if n == 0:
        return None
    r = np.asarray(run_px, dtype=np.float32)
    if np.any(r <= 0):
        return None

    candidates = _valid_module_tuples(n, modules_total)
    if not candidates:
        return None

    denom = float(np.sum(r * r) + 1e-6)
    best_m: Optional[Tuple[int, ...]] = None
    best_err = float("inf")

    for m in candidates:
        mm = np.asarray(m, dtype=np.float32)
        # best scale (least squares with zero intercept)
        k = float(np.dot(r, mm) / (np.dot(mm, mm) + 1e-6))
        pred = k * mm
        err = float(np.sum((r - pred) ** 2) / denom)
        if err < best_err:
            best_err = err
            best_m = m

    if best_m is None:
        return None
    return best_m, best_err


def _find_start_positions(run_lengths: Sequence[int]) -> List[Tuple[int, int]]:
    """Return list of (pos, start_value) for plausible Code128 start patterns."""
    starts = []
    for pos in range(0, max(0, len(run_lengths) - 6)):
        best = _best_modules_for_runs(tuple(run_lengths[pos : pos + 6]), 11)
        if best is None:
            continue
        q, err = best
        # start should be crisp
        if err > 0.25:
            continue
        v = _RUNS_TO_VALUE.get(q)
        if v in (103, 104, 105):  # StartA/B/C
            starts.append((pos, v))
    return starts


def _decode_from_runs(
    run_lengths: Sequence[int], direction: str
) -> Optional[Code128Result]:
    # Ensure we start with a bar run (black). If not, skip first white run.
    if not run_lengths:
        return None

    candidates = _find_start_positions(run_lengths)
    if not candidates:
        return None

    for start_pos, start_value in candidates:
        codes: List[int] = [start_value]
        pos = start_pos + 6

        while pos + 6 <= len(run_lengths):
            best = _best_modules_for_runs(tuple(run_lengths[pos : pos + 6]), 11)
            if best is None:
                break
            q, err = best
            if err > 0.35:
                break
            v = _RUNS_TO_VALUE.get(q)
            if v is None:
                break
            codes.append(v)
            pos += 6

            # stop is not in CODE128_BITS table; attempt match as 7-run stop.
            if pos + 7 <= len(run_lengths):
                best_stop = _best_modules_for_runs(
                    tuple(run_lengths[pos : pos + 7]), 13
                )
                if best_stop is not None:
                    q_stop, err_stop = best_stop
                else:
                    q_stop, err_stop = None, 999.0
                if q_stop == CODE128_STOP_RUNS and err_stop <= 0.45:
                    # codes currently includes start..checksum
                    if len(codes) < 3:
                        break
                    checksum = codes[-1]
                    payload_codes = codes[1:-1]
                    if _validate_checksum(start_value, payload_codes, checksum):
                        text = _codes_to_text(start_value, payload_codes)
                        if text is None:
                            break
                        return Code128Result(
                            text=text,
                            codes=tuple(codes),
                            start_value=start_value,
                            checksum_value=checksum,
                            direction=direction,
                        )
                    break
        # try next candidate
    return None


def _validate_checksum(
    start_value: int, payload_codes: Sequence[int], checksum_value: int
) -> bool:
    s = start_value
    for i, c in enumerate(payload_codes, start=1):
        s += i * c
    return (s % 103) == checksum_value


def _codes_to_text(start_value: int, payload_codes: Sequence[int]) -> Optional[str]:
    # Code sets
    if start_value == 103:
        code_set = "A"
    elif start_value == 104:
        code_set = "B"
    elif start_value == 105:
        code_set = "C"
    else:
        return None

    out: List[str] = []
    shift_next = False

    def decode_char(current_set: str, code: int) -> str:
        if current_set == "B":
            return chr(code + 32)
        if current_set == "A":
            return chr(code + 32) if code < 64 else chr(code - 64)
        raise ValueError("bad set")

    i = 0
    while i < len(payload_codes):
        c = payload_codes[i]

        # Special codes (values 96..102)
        if c == 98:  # Shift
            shift_next = True
            i += 1
            continue
        if c == 99:  # Code C
            code_set = "C"
            i += 1
            continue
        if c == 100:  # Code B
            code_set = "B"
            i += 1
            continue
        if c == 101:  # Code A
            code_set = "A"
            i += 1
            continue
        if c in (96, 97, 102):  # FNC3/FNC2/FNC1 -> ignore but keep placeholder
            # In many applications FNC1 is GS1 separator; for this task we keep it as <FNC1>.
            out.append({96: "<FNC3>", 97: "<FNC2>", 102: "<FNC1>"}[c])
            i += 1
            continue

        if code_set == "C":
            if 0 <= c <= 99:
                out.append(f"{c:02d}")
            else:
                return None
            i += 1
            continue

        # Code set A/B normal data 0..95
        if not (0 <= c <= 95):
            return None

        use_set = code_set
        if shift_next:
            use_set = "A" if code_set == "B" else "B"
            shift_next = False
        out.append(decode_char(use_set, c))
        i += 1

    return "".join(out)


def decode_code128_from_binary_rows(
    binary: np.ndarray,
    *,
    max_rows: int = 21,
    row_band: float = 0.35,
    min_runs: int = 30,
    band_half: int = 2,
) -> Optional[Code128Result]:
    """
    Decode Code128 from a binarized image (uint8 0/255 or bool), where bars are dark.
    Tries multiple horizontal scanlines and both directions.
    """
    if binary.ndim != 2:
        raise ValueError("binary must be 2D")

    if binary.dtype != np.uint8:
        b = (binary > 0).astype(np.uint8) * 255
    else:
        b = binary

    h, w = b.shape
    y0 = int(h * (0.5 - row_band / 2))
    y1 = int(h * (0.5 + row_band / 2))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 <= y0:
        y0, y1 = 0, h - 1

    rows = np.linspace(y0, y1, max_rows).astype(int).tolist()
    rows = sorted(set(rows))

    def runs_for_row_with_polarity(y: int, *, black_is_dark: bool) -> List[int]:
        """
        Build run-lengths for a scanline.
        If black_is_dark=True, bars are pixels <128. Otherwise bars are pixels >128.
        """
        y0b = max(0, y - band_half)
        y1b = min(h - 1, y + band_half)
        # average several rows to reduce speckle/broken bars
        row = np.mean(b[y0b : y1b + 1, :].astype(np.float32), axis=0)
        if black_is_dark:
            black = (row < 128.0).astype(np.uint8)
        else:
            black = (row > 128.0).astype(np.uint8)

        idx = np.flatnonzero(black)
        if idx.size == 0:
            return []
        a, z = int(idx[0]), int(idx[-1])
        sig = black[a : z + 1]

        runs: List[int] = []
        cur = int(sig[0])
        n = 1
        for v in sig[1:]:
            vv = int(v)
            if vv == cur:
                n += 1
            else:
                runs.append(n)
                cur = vv
                n = 1
        runs.append(n)

        # Ensure starts with a black run (bar)
        if int(sig[0]) == 0 and runs:
            runs = runs[1:]
        return runs

    best: Optional[Code128Result] = None
    for y in rows:
        for black_is_dark in (True, False):
            runs = runs_for_row_with_polarity(y, black_is_dark=black_is_dark)
            if len(runs) < min_runs:
                continue

            r1 = _decode_from_runs(runs, direction="lr")
            if r1 is not None:
                return r1
            r2 = _decode_from_runs(list(reversed(runs)), direction="rl")
            if r2 is not None:
                return r2
            best = best or r1 or r2

    return best
