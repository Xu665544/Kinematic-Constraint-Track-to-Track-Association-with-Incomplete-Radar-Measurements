"""
ttta_adsb_radar_kcdtw.py

TTTA (track-to-track association) between incomplete radar measurements and complete ADS-B tracks
using Mahalanobis-DTW (baseline) and KC-DTW (proposed kinematic-constraint DTW).

Key points (aligned with your paper assumptions):
- ADS-B measurements are complete.
- Radar measurements are incomplete: any subset of {r, theta, phi, v, psi} may be missing (NaN).
- We use per-sample selection (masking) to compute a "reduced-dimension" Mahalanobis local cost.
- KC-DTW adds a kinematic residual term based on first-order differences (per-second),
  which keeps discriminability when range r is missing.

Expected CSV formats
--------------------
ADS-B CSV: columns (at minimum)
    time, ID, lat, lon, alt, course, vel
Radar CSV: columns (at minimum)
    time, ID, r_m, theta_deg, phi_deg
Optional radar columns:
    course_deg, speed_kt

You can override filename patterns in run_ttta_for_scene().

Dependencies: numpy, pandas, scipy

Author: generated for your paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ------------------------- helpers: angles & time -------------------------

def wrap_angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Return (a - b) wrapped to (-180, 180]."""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    # Ensure 180 maps to -180 consistently (optional)
    d[d == 180.0] = -180.0
    return d


def _to_datetime_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    # Try common formats; fall back to pandas parser
    return pd.to_datetime(s.astype(str), errors="coerce")


def time_to_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert time array to seconds since start (relative time, float).
    Supports numpy datetime64, pandas Timestamp, numeric, or parseable strings.
    """
    if np.issubdtype(t.dtype, np.datetime64):
        t_ns = t.astype("datetime64[ns]").astype(np.int64)
        t0 = t_ns[0]
        return ((t_ns - t0) / 1e9).astype(float)

    if np.issubdtype(t.dtype, np.number):
        t0 = float(t[0])
        return (t.astype(float) - t0).astype(float)

    # object / string
    tt = pd.to_datetime(pd.Series(t), errors="coerce").to_numpy(dtype="datetime64[ns]")
    t_ns = tt.astype(np.int64)
    t0 = t_ns[0]
    return ((t_ns - t0) / 1e9).astype(float)


def time_to_epoch_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert time array to epoch seconds (absolute time, float).
    For datetime64 -> seconds since UNIX epoch.
    For numeric -> return as float.
    For strings/objects -> parse to datetime then convert.
    """
    if np.issubdtype(t.dtype, np.datetime64):
        t_ns = t.astype("datetime64[ns]").astype(np.int64)
        return (t_ns / 1e9).astype(float)

    if np.issubdtype(t.dtype, np.number):
        return t.astype(float)

    tt = pd.to_datetime(pd.Series(t), errors="coerce").to_numpy(dtype="datetime64[ns]")
    t_ns = tt.astype(np.int64)
    return (t_ns / 1e9).astype(float)


# ------------------------- coordinate conversions -------------------------

def lla_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray,
               radar_lla: Tuple[float, float, float]) -> np.ndarray:
    """
    Convert LLA to local ENU (m) using small-area approximation.
    This is sufficient for your scenarios (< few hundred km) and consistent with your existing pipeline.
    """
    lat0, lon0, h0 = radar_lla
    R = 6371000.0  # mean earth radius (m)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)

    dlat = lat - lat0r
    dlon = lon - lon0r

    east = R * np.cos(lat0r) * dlon
    north = R * dlat
    up = alt_m - h0
    return np.column_stack([east, north, up])


def enu_to_radar_meas(enu: np.ndarray) -> np.ndarray:
    """
    ENU -> (r, theta, phi)
    - r: slant range (m)
    - theta: azimuth (deg), North=0, clockwise positive
    - phi: elevation (deg), [-90, 90]
    """
    e = enu[:, 0]
    n = enu[:, 1]
    u = enu[:, 2]
    r = np.sqrt(e**2 + n**2 + u**2)

    theta = np.rad2deg(np.arctan2(e, n)) % 360.0  # atan2(E, N)
    ground = np.sqrt(e**2 + n**2)
    phi = np.rad2deg(np.arctan2(u, ground))
    phi = np.clip(phi, -90.0, 90.0)

    return np.column_stack([r, theta, phi])


# ------------------------- track containers -------------------------

@dataclass
class Track:
    track_id: str
    t: np.ndarray               # (n,) time, numpy array
    z: np.ndarray               # (n, D) measurement, NaN allowed (radar incomplete)
    dz: np.ndarray              # (n, D) first diff per second, NaN allowed
    # optionally store raw indices etc if needed


def compute_diff_per_second(z: np.ndarray, t: np.ndarray, angle_dims: List[int]) -> np.ndarray:
    """
    First-order difference per second for each dimension.
    If dt<=0 or either sample missing, output NaN at that index.
    """
    n, d = z.shape
    dz = np.full_like(z, np.nan, dtype=float)
    ts = time_to_seconds(t)
    dt = np.diff(ts, prepend=np.nan)  # dt[0]=nan

    for k in range(1, n):
        if not np.isfinite(dt[k]) or dt[k] <= 0:
            continue
        for j in range(d):
            a = z[k, j]
            b = z[k - 1, j]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if j in angle_dims:
                dv = wrap_angle_diff_deg(np.array([a]), np.array([b]))[0] / dt[k]
            else:
                dv = (a - b) / dt[k]
            dz[k, j] = dv
    return dz


# ------------------------- local cost: Mahalanobis + KC -------------------------

@dataclass
class CostParams:
    # Measurement sigmas
    sigma_r: float = 200.0
    sigma_theta_deg: float = 1.0
    sigma_phi_deg: float = 1.0
    sigma_v: float = 20.0          # knots
    sigma_psi_deg: float = 5.0     # degrees

    # Kinematic (first-diff) sigmas
    sigma_dr: float = 50.0         # m/s (range-rate)
    sigma_dtheta: float = 0.2      # deg/s
    sigma_dphi: float = 0.2        # deg/s
    sigma_dv: float = 5.0          # kt/s
    sigma_dpsi: float = 1.0        # deg/s

    # KC weight
    kc_weight: float = 1.0

    # Time gating (optional). If set, pairs with |t_r - t_a| > gate are penalized by +inf.
    time_gate_sec: Optional[float] = None


def _mahalanobis_masked(res: np.ndarray, sig: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute masked Mahalanobis distance (squared form):
        d = sum( (res_i / sig_i)^2 ) over valid i
    """
    if mask.sum() == 0:
        return float("inf")
    v = res[mask] / sig[mask]
    return float(np.dot(v, v) / mask.sum())  # normalized by dim count (keeps scale stable)


def local_cost_kcdtw(
    z_r: np.ndarray, z_a: np.ndarray,
    dz_r: np.ndarray, dz_a: np.ndarray,
    t_r: float, t_a: float,
    params: CostParams,
    use_kc: bool = True
) -> float:
    """
    Combined local cost:
        d = d_meas + kc_weight * d_kc

    d_meas is reduced-dimension Mahalanobis on [r, theta, phi, v, psi].
    d_kc   is reduced-dimension Mahalanobis on first differences per second.
    """
    # time gate
    if params.time_gate_sec is not None:
        if abs(t_r - t_a) > params.time_gate_sec:
            return float("inf")

    # dims: 0=r, 1=theta, 2=phi, 3=v, 4=psi
    sig_meas = np.array([params.sigma_r, params.sigma_theta_deg, params.sigma_phi_deg,
                         params.sigma_v, params.sigma_psi_deg], dtype=float)

    # residuals (handle angles)
    res = np.full(5, np.nan, dtype=float)
    # r
    res[0] = z_r[0] - z_a[0]
    # theta
    res[1] = wrap_angle_diff_deg(np.array([z_r[1]]), np.array([z_a[1]]))[0]
    # phi
    res[2] = z_r[2] - z_a[2]
    # v
    res[3] = z_r[3] - z_a[3]
    # psi
    res[4] = wrap_angle_diff_deg(np.array([z_r[4]]), np.array([z_a[4]]))[0]

    mask_meas = np.isfinite(z_r) & np.isfinite(z_a)
    d_meas = _mahalanobis_masked(res, sig_meas, mask_meas)

    if (not use_kc) or (params.kc_weight <= 0):
        return d_meas

    # KC term: first difference per second
    sig_kc = np.array([params.sigma_dr, params.sigma_dtheta, params.sigma_dphi,
                       params.sigma_dv, params.sigma_dpsi], dtype=float)

    res_kc = np.full(5, np.nan, dtype=float)
    res_kc[0] = dz_r[0] - dz_a[0]
    res_kc[1] = dz_r[1] - dz_a[1]
    res_kc[2] = dz_r[2] - dz_a[2]
    res_kc[3] = dz_r[3] - dz_a[3]
    res_kc[4] = dz_r[4] - dz_a[4]

    # Note: dtheta/dpsi already wrapped inside compute_diff_per_second()
    mask_kc = np.isfinite(dz_r) & np.isfinite(dz_a)
    # If no kinematic overlap, do not add penalty (KC term is "inactive")
    if mask_kc.sum() == 0:
        return d_meas

    d_kc = _mahalanobis_masked(res_kc, sig_kc, mask_kc)
    return d_meas + params.kc_weight * d_kc


# ------------------------- DTW (given local costs) -------------------------

def dtw_cost_from_delta(delta: np.ndarray) -> float:
    """
    Standard DTW with step set {(1,0),(0,1),(1,1)}.
    Return normalized DTW cost (average along the optimal warping path).
    """
    n, m = delta.shape
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    L = np.zeros((n + 1, m + 1), dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = delta[i - 1, j - 1]
            if not np.isfinite(c):
                continue
            candidates = [
                (D[i - 1, j], L[i - 1, j]),
                (D[i, j - 1], L[i, j - 1]),
                (D[i - 1, j - 1], L[i - 1, j - 1]),
            ]
            idx = int(np.argmin([x[0] for x in candidates]))
            bestD, bestL = candidates[idx]
            if np.isfinite(bestD):
                D[i, j] = c + bestD
                L[i, j] = bestL + 1.0

    if not np.isfinite(D[n, m]) or L[n, m] <= 0:
        return float("inf")
    return float(D[n, m] / L[n, m])


def dtw_debug_path(delta: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    DTW with backtracking. Return (normalized_cost, path) where path is Kx2 indices (i,j).
    """
    n, m = delta.shape
    D = np.full((n + 1, m + 1), np.inf, dtype=float)
    ptr = np.full((n + 1, m + 1, 2), -1, dtype=int)
    L = np.zeros((n + 1, m + 1), dtype=float)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = delta[i - 1, j - 1]
            if not np.isfinite(c):
                continue
            candidates = [
                (D[i - 1, j], i - 1, j),
                (D[i, j - 1], i, j - 1),
                (D[i - 1, j - 1], i - 1, j - 1),
            ]
            best = min(candidates, key=lambda x: x[0])
            if np.isfinite(best[0]):
                D[i, j] = c + best[0]
                ptr[i, j] = [best[1], best[2]]
                # path length
                if best[1] == i - 1 and best[2] == j - 1:
                    L[i, j] = L[i - 1, j - 1] + 1.0
                elif best[1] == i - 1:
                    L[i, j] = L[i - 1, j] + 1.0
                else:
                    L[i, j] = L[i, j - 1] + 1.0

    if not np.isfinite(D[n, m]) or L[n, m] <= 0:
        return float("inf"), np.zeros((0, 2), dtype=int)

    # backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        pi, pj = ptr[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = int(pi), int(pj)
    path.reverse()
    return float(D[n, m] / L[n, m]), np.asarray(path, dtype=int)


# ------------------------- track building -------------------------

def build_adsb_tracks(adsb_csv: str, radar_lla: Tuple[float, float, float]) -> Tuple[List[str], List[Track]]:
    df = pd.read_csv(adsb_csv)
    if "ID" not in df.columns:
        raise ValueError(f"ADS-B file missing ID column: {adsb_csv}")

    # Ensure time exists
    if "time" in df.columns:
        df["time"] = _to_datetime_series(df["time"])
    else:
        df["time"] = pd.RangeIndex(len(df)).astype(float)

    # Required columns for geometry
    for col in ("lat", "lon", "alt"):
        if col not in df.columns:
            raise ValueError(f"ADS-B file missing '{col}' column: {adsb_csv}")

    # Motion features (complete by your assumption)
    if "course" not in df.columns or "vel" not in df.columns:
        raise ValueError(f"ADS-B file must contain 'course' and 'vel' columns: {adsb_csv}")

    df["ID"] = df["ID"].astype(str).str.strip()
    df = df.sort_values(["ID", "time"], kind="mergesort")

    tracks: List[Track] = []
    ids: List[str] = []

    for tid, g in df.groupby("ID", sort=False):
        t = g["time"].to_numpy()
        lat = g["lat"].to_numpy(dtype=float)
        lon = g["lon"].to_numpy(dtype=float)
        alt = g["alt"].to_numpy(dtype=float)

        course = pd.to_numeric(g["course"], errors="coerce").to_numpy(dtype=float)
        vel = pd.to_numeric(g["vel"], errors="coerce").to_numpy(dtype=float)

        # Build radar-domain measurement of ADS-B: [r,theta,phi,v,psi]
        enu = lla_to_enu(lat, lon, alt, radar_lla)
        rthphi = enu_to_radar_meas(enu)

        z = np.column_stack([rthphi[:, 0], rthphi[:, 1], rthphi[:, 2], vel, course])

        # ADS-B complete: still, remove rows with any nonfinite in required geometry
        ok = np.isfinite(rthphi).all(axis=1)
        t = t[ok]
        z = z[ok]
        if len(z) < 2:
            continue

        dz = compute_diff_per_second(z, t, angle_dims=[1, 4])  # theta, psi
        tracks.append(Track(track_id=str(tid), t=t, z=z, dz=dz))
        ids.append(str(tid))

    return ids, tracks


def build_radar_tracks(radar_csv: str) -> Tuple[List[str], List[Track]]:
    df = pd.read_csv(radar_csv)
    if "ID" not in df.columns:
        raise ValueError(f"Radar file missing ID column: {radar_csv}")

    if "time" in df.columns:
        df["time"] = _to_datetime_series(df["time"])
    else:
        df["time"] = pd.RangeIndex(len(df)).astype(float)

    # Accept both naming styles
    col_r = "r_m" if "r_m" in df.columns else ("range_m" if "range_m" in df.columns else None)
    col_th = "theta_deg" if "theta_deg" in df.columns else ("az_deg" if "az_deg" in df.columns else None)
    col_ph = "phi_deg" if "phi_deg" in df.columns else ("el_deg" if "el_deg" in df.columns else None)
    if col_r is None or col_th is None or col_ph is None:
        raise ValueError(f"Radar file must contain r_m/theta_deg/phi_deg (or aliases): {radar_csv}")

    col_psi = "course_deg" if "course_deg" in df.columns else ("course" if "course" in df.columns else None)
    col_v = "speed_kt" if "speed_kt" in df.columns else ("vel" if "vel" in df.columns else None)

    df["ID"] = df["ID"].astype(str).str.strip()
    df = df.sort_values(["ID", "time"], kind="mergesort")

    tracks: List[Track] = []
    ids: List[str] = []

    for tid, g in df.groupby("ID", sort=False):
        t = g["time"].to_numpy()
        r = pd.to_numeric(g[col_r], errors="coerce").to_numpy(dtype=float)
        th = pd.to_numeric(g[col_th], errors="coerce").to_numpy(dtype=float)
        ph = pd.to_numeric(g[col_ph], errors="coerce").to_numpy(dtype=float)

        if col_v is None:
            v = np.full_like(r, np.nan, dtype=float)
        else:
            v = pd.to_numeric(g[col_v], errors="coerce").to_numpy(dtype=float)

        if col_psi is None:
            psi = np.full_like(r, np.nan, dtype=float)
        else:
            psi = pd.to_numeric(g[col_psi], errors="coerce").to_numpy(dtype=float)

        # Normalize angles ranges (optional)
        th = np.mod(th, 360.0)
        psi = np.mod(psi, 360.0)

        z = np.column_stack([r, th, ph, v, psi])
        # Keep even if missing dims; but require at least theta&phi or r&theta etc? keep if any finite.
        ok = np.isfinite(z).any(axis=1)
        t = t[ok]
        z = z[ok]
        if len(z) < 2:
            continue

        dz = compute_diff_per_second(z, t, angle_dims=[1, 4])  # theta, psi
        tracks.append(Track(track_id=str(tid), t=t, z=z, dz=dz))
        ids.append(str(tid))

    return ids, tracks


# ------------------------- TTTA driver -------------------------

def compute_cost_matrix(
    radar_tracks: List[Track],
    adsb_tracks: List[Track],
    params: CostParams,
    use_kc: bool = True
) -> np.ndarray:
    """
    C[i,j] = DTW cost between radar track i and ADS-B track j.
    """
    nR = len(radar_tracks)
    nA = len(adsb_tracks)
    C = np.full((nR, nA), np.inf, dtype=float)

    for i, tr in enumerate(radar_tracks):
        tr_t = time_to_seconds(tr.t)
        tr_te = time_to_epoch_seconds(tr.t)
        for j, ta in enumerate(adsb_tracks):
            ta_t = time_to_seconds(ta.t)
            ta_te = time_to_epoch_seconds(ta.t)

            # local delta matrix
            delta = np.full((len(tr_t), len(ta_t)), np.inf, dtype=float)
            for ii in range(len(tr_t)):
                for jj in range(len(ta_t)):
                    delta[ii, jj] = local_cost_kcdtw(
                        tr.z[ii], ta.z[jj],
                        tr.dz[ii], ta.dz[jj],
                        float(tr_te[ii]), float(ta_te[jj]),
                        params=params,
                        use_kc=use_kc
                    )
            C[i, j] = dtw_cost_from_delta(delta)

    return C


def solve_assignment(C: np.ndarray) -> List[Tuple[int, int]]:
    """
    Global 1-1 assignment with Hungarian algorithm.
    Rows: radar IDs, Cols: ADS-B IDs.
    """
    # Replace inf with a large number for assignment stability
    finite = np.isfinite(C)
    if not finite.any():
        return []
    big = np.nanmax(C[finite]) * 10.0 + 1.0
    C2 = C.copy()
    C2[~finite] = big
    r_ind, c_ind = linear_sum_assignment(C2)
    return list(zip(r_ind.tolist(), c_ind.tolist()))


def run_ttta_for_scene(
    scene_id: int,
    level: int,
    base_dir: str,
    radar_lla: Tuple[float, float, float],
    params: Optional[CostParams] = None,
    use_kc: bool = True,
    adsb_pattern: str = "scene{sid}_ads-b_{lv}.csv",
    radar_pattern: str = "scene{sid}_radar_{lv}_miss.csv"
) -> Tuple[List[Tuple[str, str, float]], np.ndarray, List[Tuple[int, int]], List[str], List[str]]:
    """
    Main entry:
        - build tracks
        - compute cost matrix
        - solve assignment

    Returns:
        pairs, C, matches, radar_ids, adsb_ids (for downstream logging/plots)
    """
    if params is None:
        params = CostParams()

    adsb_csv = os.path.join(base_dir, adsb_pattern.format(sid=scene_id, lv=level))
    radar_csv = os.path.join(base_dir, radar_pattern.format(sid=scene_id, lv=level))

    adsb_ids, adsb_tracks = build_adsb_tracks(adsb_csv, radar_lla)
    radar_ids, radar_tracks = build_radar_tracks(radar_csv)

    C = compute_cost_matrix(radar_tracks, adsb_tracks, params=params, use_kc=use_kc)
    matches = solve_assignment(C)

    pairs = [(radar_ids[i], adsb_ids[j], float(C[i, j])) for (i, j) in matches]
    return pairs, C, matches, radar_ids, adsb_ids


# ------------------------- optional: debug for one pair -------------------------

def debug_one_pair(
    radar_track: Track,
    adsb_track: Track,
    params: Optional[CostParams] = None,
    use_kc: bool = True
) -> Dict[str, np.ndarray]:
    """
    Produce delta matrix + warping path for a specific pair; useful for paper figures.
    """
    if params is None:
        params = CostParams()

    tr_t = time_to_seconds(radar_track.t)
    ta_t = time_to_seconds(adsb_track.t)
    tr_te = time_to_epoch_seconds(radar_track.t)
    ta_te = time_to_epoch_seconds(adsb_track.t)
    delta = np.full((len(tr_t), len(ta_t)), np.inf, dtype=float)
    for ii in range(len(tr_t)):
        for jj in range(len(ta_t)):
            delta[ii, jj] = local_cost_kcdtw(
                radar_track.z[ii], adsb_track.z[jj],
                radar_track.dz[ii], adsb_track.dz[jj],
                float(tr_te[ii]), float(ta_te[jj]),
                params=params,
                use_kc=use_kc
            )
    cost, path = dtw_debug_path(delta)
    return {"delta": delta, "path": path, "cost": np.array([cost], dtype=float)}
