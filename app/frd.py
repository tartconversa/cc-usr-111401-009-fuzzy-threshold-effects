"""模糊断点回归（FRD）：门槛附近资格变动对领取与经营结果的局部影响。

设计约定（与 docs/data-contract.md 一致）：
- 运行变量只使用干预前评分 score；门槛取 metadata.coff（单一门槛，不混用后补评分或其他门槛）。
- 资格 Z = 1{score >= cutoff}，与实际领取 received 分开；估计量为门槛处依从者的局部效应
  LATE = 结果跳跃 tau_y / 领取跳跃 tau_d。
- 同主体多期结果按相对 decision_day 的窗口取均值，每主体计一次。
- 结果跳跃与领取跳跃在同一加权矩体系内联合估计，协方差取联合三明治估计的子块，
  不做“两个数相除再套正态区间”的弱识别推断。

本模块仅使用 Python 标准库，不改变 app.service / app.baseline 的既有接口与语义。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .catalog import load
from .baseline import analyze
from .service import SCHEMAS

# ---------------------------------------------------------------------------
# 基础数值工具（标准库）
# ---------------------------------------------------------------------------

def _mat(n, m, fill=0.0):
    return [[fill for _ in range(m)] for _ in range(n)]

def _solve(A, b):
    """高斯部分主元消去，解一般线性方程组；奇异时抛出 ValueError。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError('设计矩阵秩不足，支撑点过少')
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / piv
            if factor:
                for k in range(col, n + 1):
                    M[r][k] -= factor * M[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        residual = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = residual / M[i][i]
    return x

def _sym_inverse(A):
    """对称正定矩阵求逆（用于面包矩阵 A_hat^{-1}）。"""
    n = len(A)
    inv = [_solve(A, [1.0 if i == j else 0.0 for j in range(n)]) for i in range(n)]
    return [[inv[j][i] for j in range(n)] for i in range(n)]  # 转置为按行存放

def _block_diag(blocks):
    rows = sum(b[0] for b in blocks)
    cols = sum(b[1] for b in blocks)
    out = _mat(rows, cols)
    r0 = c0 = 0
    for br, bc, block in blocks:
        for i in range(br):
            for j in range(bc):
                out[r0 + i][c0 + j] = block[i][j]
        r0 += br
        c0 += bc
    return out

def _wls(X, w, y):
    XtW = [[w[i] * X[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    XtWX = [[sum(XtW[i][j] * X[i][k] for i in range(len(X))) for k in range(len(X[0]))]
            for j in range(len(X[0]))]
    XtWy = [sum(XtW[i][j] * y[i] for i in range(len(X))) for j in range(len(X[0]))]
    return _solve(XtWX, XtWy)

def _norm_q(p):
    """标准正态分位数（Acklam 算法），用于区间端点。"""
    if not 0.0 < p < 1.0:
        raise ValueError('分位数概率越界')
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

def _chi2_q1(p):
    """自由度 1 的卡方分位数，即标准正态分位数的平方。"""
    z = _norm_q(0.5 + p / 2)
    return z * z

# ---------------------------------------------------------------------------
# 局部设计与联合加权矩估计
# ---------------------------------------------------------------------------

def _kernel_weight(distance, bandwidth, kernel):
    u = abs(distance) / bandwidth
    if u > 1.0:
        return 0.0
    return 1.0 - u if kernel == 'triangular' else 1.0

def _design_row(z, side, degree):
    row = [1.0, z]
    if degree >= 2:
        row.append(z * z)
    row.append(float(side))
    if degree >= 1:
        row.append(z * float(side))
    if degree >= 2:
        row.append(z * z * float(side))
    return row

def _n_params(degree):
    return 2 * (degree + 1)

def _joint_fit(frames, bandwidths, cutoff, degree=1, kernel='triangular'):
    """结果方程与领取方程的主体级联合加权矩估计，离散评分按评分点聚类。

    两个方程共享同一批主体（顺序一致），各自携带带宽；矩条件按主体堆叠为
    psi_i = [w_y X_iy e_iy ; w_d X_id e_id]。面包为分块对角（矩条件的条件均值），
    肉矩阵先在评分点（簇）内聚合、再跨簇求和，保留同簇相关与跨方程协方差——
    离散运行变量下同评分点的多个主体不是独立观测，直接按主体估三明治会低估方差
    （Lee & Card 2008）。簇数偏少时代入 G/(G-1) 小样本校正。
    """
    k = _n_params(degree)
    prepared = []
    for frame, bw in zip(frames, bandwidths):
        r, z, y, sc = [], [], [], []
        for row in frame:
            w = _kernel_weight(row['score'] - cutoff, bw, kernel)
            if w <= 0.0:
                continue
            r.append(w)
            z.append(row['score'] - cutoff)
            y.append(row['y'])
            sc.append(row['score'])
        if len(r) < k + 2:
            raise ValueError('带宽内观测不足，无法识别局部模型')
        X = [_design_row(zi, 1 if zi >= 0 else 0, degree) for zi in z]
        beta = _wls(X, r, y)
        resid = [y[i] - sum(X[i][j] * beta[j] for j in range(k)) for i in range(len(r))]
        G = [[sum(r[i] * X[i][a] * X[i][b] for i in range(len(r))) for b in range(k)]
             for a in range(k)]
        prepared.append({'X': X, 'r': r, 'resid': resid, 'beta': beta, 'G': G,
                         'n': len(r), 'scores': sc})
    if prepared[0]['n'] != prepared[1]['n']:
        raise ValueError('结果与领取方程主体数不一致，无法联合估计')
    nobs = prepared[0]['n']
    # 两个方程主体对齐，评分一致；以评分点为聚类键
    cluster_keys = []
    cluster_index = {}
    for s in prepared[0]['scores']:
        if s not in cluster_index:
            cluster_index[s] = len(cluster_keys)
            cluster_keys.append(s)
    g_clusters = len(cluster_keys)
    if g_clusters < k + 1:
        raise ValueError(f'带宽内评分点（簇）仅 {g_clusters} 个，少于参数所需，局部模型不可识别')
    total_p = 2 * k
    Gfull = _block_diag([(k, k, prepared[0]['G']), (k, k, prepared[1]['G'])])
    bread = _sym_inverse(Gfull)
    # 簇级聚合的联合肉矩阵：先在每个评分点内对主体得分求和，再取簇和的外积，
    # 含跨方程块；同簇主体的共同冲击由此进入，而不会被当作独立观测平均掉。
    cluster_sums = [[[0.0] * k, [0.0] * k] for _ in range(g_clusters)]
    for i in range(nobs):
        cg = cluster_index[prepared[0]['scores'][i]]
        for j in range(k):
            cluster_sums[cg][0][j] += prepared[0]['r'][i] * prepared[0]['X'][i][j] * prepared[0]['resid'][i]
            cluster_sums[cg][1][j] += prepared[1]['r'][i] * prepared[1]['X'][i][j] * prepared[1]['resid'][i]
    meat = _mat(total_p, total_p)
    for cg in range(g_clusters):
        u0, u1 = cluster_sums[cg][0], cluster_sums[cg][1]
        for a in range(k):
            for b in range(k):
                meat[a][b] += u0[a] * u0[b]
                meat[a][k + b] += u0[a] * u1[b]
                meat[k + a][b] += u1[a] * u0[b]
                meat[k + a][k + b] += u1[a] * u1[b]
    # Lee–Card 小样本校正 G/(G-1)
    correction = g_clusters / max(g_clusters - 1, 1)
    V = [[correction * sum(bread[i][m] * meat[m][n] * bread[j][n]
                           for m in range(total_p) for n in range(total_p))
          for j in range(total_p)] for i in range(total_p)]
    beta_full = prepared[0]['beta'] + prepared[1]['beta']
    iy, idd = degree + 1, k + degree + 1
    Vj = [[V[iy][iy], V[iy][idd]], [V[idd][iy], V[idd][idd]]]
    return {'beta': beta_full, 'V': V, 'jump_index': [iy, idd],
            'tau_y': beta_full[iy], 'tau_d': beta_full[idd], 'V_jump': Vj,
            'n': nobs, 'n_clusters': g_clusters}

def _single_jump(frame, bandwidth, cutoff, degree, kernel):
    """单方程局部加权跳跃估计；离散评分按评分点聚类给标准误。"""
    r, z, y, sides, sc = [], [], [], [], []
    for row in frame:
        w = _kernel_weight(row['score'] - cutoff, bandwidth, kernel)
        if w <= 0.0:
            continue
        r.append(w)
        z.append(row['score'] - cutoff)
        y.append(row['y'])
        sides.append(1 if row['score'] >= cutoff else 0)
        sc.append(row['score'])
    k = _n_params(degree)
    if len(r) < k + 2:
        raise ValueError('带宽内观测不足')
    X = [_design_row(zi, sides[i], degree) for i, zi in enumerate(z)]
    beta = _wls(X, r, y)
    resid = [y[i] - sum(X[i][j] * beta[j] for j in range(k)) for i in range(len(r))]
    G = [[sum(r[i] * X[i][a] * X[i][b] for i in range(len(r))) for b in range(k)]
         for a in range(k)]
    bread = _sym_inverse(G)
    clusters = {}
    for i, s in enumerate(sc):
        clusters.setdefault(s, []).append(i)
    g = len(clusters)
    if g < k + 1:
        raise ValueError('带宽内评分点（簇）不足，诊断跳跃不可估计')
    meat = _mat(k, k)
    for idxs in clusters.values():
        u = [sum(r[i] * X[i][j] * resid[i] for i in idxs) for j in range(k)]
        for a in range(k):
            for b in range(k):
                meat[a][b] += u[a] * u[b]
    correction = g / max(g - 1, 1)
    V = [[correction * sum(bread[a][m] * meat[m][n] * bread[b][n]
                           for m in range(k) for n in range(k))
          for b in range(k)] for a in range(k)]
    j = degree + 1
    return {'jump': beta[j], 'se': math.sqrt(max(V[j][j], 0.0)),
            'n_left': sum(1 for s in sides if s == 0),
            'n_right': sum(1 for s in sides if s == 1),
            'n_clusters': g}

# ---------------------------------------------------------------------------
# Imbens–Kalyanaraman (2012) 局部线性带宽
# ---------------------------------------------------------------------------

IK_CONSTANTS = {1: 1.715, 2: 2.682}  # 三角核、局部 p 阶（IK 表）

def _ik_bandwidth(scores, values, cutoff, degree=1):
    """IK 2012 带宽；任一步骤无法支撑时回退到评分跨度的一半。"""
    pairs = sorted(zip(scores, values))
    r_all = [s - cutoff for s, _ in pairs]
    span = max(max(r_all), -min(r_all))
    if span <= 0:
        raise ValueError('评分无变异')
    try:
        # 1) 全局三次样条近似：两侧各拟合三次多项式，取门槛处二阶导数
        m2 = []
        for side in (-1, 1):
            data = [(r, v) for r, v in zip(r_all, [v for _, v in pairs])
                    if (r < 0 if side == -1 else r >= 0)]
            if len(data) < 5:
                raise ValueError('单侧样本不足以估计曲率')
            rs = [d[0] for d in data]
            vs = [d[1] for d in data]
            X = [[1.0, r, r * r, r ** 3] for r in rs]
            beta = _wls(X, [1.0] * len(rs), vs)
            m2.append(2.0 * beta[2])  # 二阶导数 = 2! * 二次项系数
        curvature = sum(m2) / 2
        # 2) 等距粗分箱（箱宽 = 2*跨度/10 与最小评分间距的较大者）
        gaps = [r_all[i + 1] - r_all[i] for i in range(len(r_all) - 1) if r_all[i + 1] > r_all[i]]
        binwidth = max(2 * span / 10, min(gaps) if gaps else span / 10)
        bins = {}
        for r, v in zip(r_all, [v for _, v in pairs]):
            key = int(math.floor(r / binwidth))
            bins.setdefault(key, []).append(v)
        rbar, ybar, cbar = [], [], []
        for key, vals in sorted(bins.items()):
            rbar.append((key + 0.5) * binwidth)
            ybar.append(sum(vals) / len(vals))
            cbar.append(len(vals))
        if len(rbar) < 6:
            raise ValueError('分箱数不足')
        Xb = [[1.0, r, r * r, r ** 3] for r in rbar]
        bb = _wls(Xb, [float(c) for c in cbar], ybar)
        left_idx = [i for i, r in enumerate(rbar) if r < 0]
        right_idx = [i for i, r in enumerate(rbar) if r >= 0]
        if not left_idx or not right_idx:
            raise ValueError('门槛两侧均需有分箱')
        # IK 方差项为个体层面条件方差 sigma^2(0)：取门槛两侧最邻近分箱的原始残差，
        # 不能直接用箱均值回归残差（那是均值的方差，会随箱内样本数缩小）。
        keys = sorted(bins)
        center_idx = (max(left_idx), min(right_idx))
        raw_resid = []
        for idx in center_idx:
            pred = sum(Xb[idx][j] * bb[j] for j in range(4))
            raw_resid.extend(v - pred for v in bins[keys[idx]])
        if len(raw_resid) < 4:
            raise ValueError('门槛邻近原始观测不足以估计条件方差')
        mean_r = sum(raw_resid) / len(raw_resid)
        sigma2 = sum((r - mean_r) ** 2 for r in raw_resid) / (len(raw_resid) - 1)
        n_local = sum(1 for r in r_all if abs(r) <= span / 2)
        # 门槛邻域经验密度：邻域宽度为 span（两侧各 span/2），单位评分长度上的观测密度
        density = max(n_local / span, 1e-9)
        denom = curvature ** 2
        if denom <= 1e-12:
            raise ValueError('曲率接近零，IK 正则项无定义')
        const = IK_CONSTANTS.get(degree, IK_CONSTANTS[1])
        h = const * (sigma2 / (density * denom)) ** 0.2
        if not math.isfinite(h) or h <= 0:
            raise ValueError('IK 带宽非有限')
        return min(h, span)
    except (ValueError, ZeroDivisionError):
        return span / 2

# ---------------------------------------------------------------------------
# 资料整理：一主体一行，多期结果按窗口取均值
# ---------------------------------------------------------------------------

def build_persons(snapshot):
    applications = snapshot.rows('applications')
    persons = {a['person'] for a in applications}
    if len(persons) != len(applications):
        raise ValueError('同主体存在多个申请，需选择资格轮次')
    cutoff = snapshot.metadata['cutoff']
    outcomes = snapshot.rows('outcomes')
    records = []
    for a in applications:
        if a['score_day'] > a['decision_day']:
            raise ValueError('评分发生在资格决定之后')
        deltas = [o['day'] - a['decision_day'] for o in outcomes if o['person'] == a['person']]
        vals = [o['value'] for o in outcomes if o['person'] == a['person']]
        records.append({'person': a['person'], 'score': a['score'],
                        'eligible': 1 if a['score'] >= cutoff else 0,
                        'received': 1 if a['received'] else 0,
                        'score_day': a['score_day'], 'decision_day': a['decision_day'],
                        'pairs': sorted(zip(deltas, vals))})
    records.sort(key=lambda d: (d['score'], d['person']))
    return records, cutoff

def _window_frame(records, window):
    left, right = window
    frame, have = [], []
    for rec in records:
        vals = [v for d, v in rec['pairs'] if left <= d < right]
        frame.append({'person': rec['person'], 'score': rec['score'],
                      'eligible': rec['eligible'], 'received': rec['received'],
                      'y': sum(vals) / len(vals) if vals else None})
        have.append(1 if vals else 0)
    return frame, have

def _complete_case(frame, outcome_key='y'):
    return [{'score': r['score'], 'y': r[outcome_key], 'received': r['received'],
             'eligible': r['eligible'], 'person': r['person']}
            for r in frame if r[outcome_key] is not None]

def _support(frame, bandwidth, cutoff):
    left = {r['score'] for r in frame if r['score'] < cutoff and abs(r['score'] - cutoff) <= bandwidth}
    right = {r['score'] for r in frame if r['score'] >= cutoff and abs(r['score'] - cutoff) <= bandwidth}
    return len(left), len(right)

# ---------------------------------------------------------------------------
# 弱识别稳健推断：Anderson–Rubin 二次型集合与 Fieller 区间
# ---------------------------------------------------------------------------

def _quadratic_set(a, b, c, critical):
    """求解 a*r^2 + b*r + c <= critical 的 r 集合（r = tau_y/tau_d）。"""
    cc = c - critical
    disc = b * b - 4 * a * cc
    if abs(a) < 1e-14:
        return {'type': 'unbounded', 'bounds': None}
    if disc < 0:
        return {'type': 'empty' if a > 0 else 'whole_real_line', 'bounds': None}
    sq = math.sqrt(disc)
    r1, r2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    lo, hi = min(r1, r2), max(r1, r2)
    if a > 0:
        return {'type': 'bounded', 'bounds': [lo, hi]}
    return {'type': 'two_unbounded_tails', 'bounds': [lo, hi]}

def _ar_set(ty, td, V, alpha):
    """Anderson–Rubin 接受集合。

    单一工具、恰好识别下，AR 二次型与 Fieller 构造在同一置信水平代数等价，
    故只报告一次并标注等价性；其集合可以有界、无界（双侧尾）、整实线或空集，
    弱识别时天然表现为宽/无界集合，无需依赖比值的正态近似。
    """
    vy, vd, cvy = V[0][0], V[1][1], V[0][1]
    crit = _chi2_q1(1 - alpha)
    ar = _quadratic_set(vd, -2 * cvy, vy, crit)
    ar['equivalence'] = '单工具恰好识别下与同水平 Fieller 集合代数等价'
    ar['critical_value'] = crit
    return ar

def _ratio_inference(ty, td, V, alpha, boot_ratios, boot_weak=None):
    se_td = math.sqrt(max(V[1][1], 0.0))
    f_stat = (td / se_td) ** 2 if se_td > 0 else None
    weak = (se_td == 0.0) or (f_stat is not None and f_stat < 10.0)
    ar = _ar_set(ty, td, V, alpha)
    width = (ar['bounds'][1] - ar['bounds'][0]) if ar['type'] == 'bounded' and ar['bounds'] else None
    out = {
        'tau_y': ty, 'tau_d': td,
        'se_tau_y': math.sqrt(max(V[0][0], 0.0)),
        'se_tau_d': se_td,
        'cov_jumps': V[0][1],
        'first_stage_z': td / se_td if se_td > 0 else None,
        'first_stage_F': f_stat,
        'weak_identification': weak,
        'wald_ratio_point_estimate': ty / td if abs(td) > 1e-12 else None,
        'wald_ratio_delta_se_reference': None,
        'wald_warning': 'Wald 比值与 delta 法标准误仅作参照；弱识别时分布非正态，主推断以 AR 集合为准',
        'anderson_rubin_95': ar,
        'ar_set_width': width,
        'bootstrap_p95': None,
        'bootstrap_weak_F_share': boot_weak,
    }
    # 仅作参照的 Wald（delta 法）标准误，弱识别时不得作为主推断
    if abs(td) > 1e-12:
        g = [1 / td, -ty / (td * td)]
        var_r = g[0] ** 2 * V[0][0] + g[1] ** 2 * V[1][1] + 2 * g[0] * g[1] * V[0][1]
        out['wald_ratio_delta_se_reference'] = math.sqrt(max(var_r, 0.0))
    if boot_ratios:
        ordered = sorted(boot_ratios)
        m = len(ordered)
        lo_idx = max(int(0.025 * (m + 1)) - 1, 0)
        hi_idx = min(int(0.975 * (m + 1)) - 1, m - 1)
        out['bootstrap_p95'] = [ordered[lo_idx], ordered[hi_idx]]
    return out

def _paired_bootstrap(frame_y, frame_d, bandwidths, cutoff, degree, kernel, reps, seed):
    """按评分点的聚类 bootstrap：门槛两侧分层，重抽整个评分点及其全部主体。

    与按评分点聚类的三明治一致，保留同评分点主体的相关性；同时记录复现样本中
    一阶段 F<10 的弱识别份额。
    """
    rng = random.Random(seed)
    # 主体在两个方程中对齐，以评分点建簇（记录两方程的行号）
    left_clusters, right_clusters = {}, {}
    for i, row in enumerate(frame_d):
        target = left_clusters if row['score'] < cutoff else right_clusters
        target.setdefault(row['score'], []).append(i)
    if not left_clusters or not right_clusters:
        return None
    left_keys, right_keys = list(left_clusters), list(right_clusters)
    ratios, failed, weak_count, total = [], 0, 0, 0
    for _ in range(reps):
        idx = []
        for keys, clusters in ((left_keys, left_clusters), (right_keys, right_clusters)):
            for _ in keys:
                key = rng.choice(keys)
                idx.extend(clusters[key])
        fy = [frame_y[i] for i in idx]
        fd = [frame_d[i] for i in idx]
        total += 1
        try:
            fit = _joint_fit([fy, fd], bandwidths, cutoff, degree, kernel)
        except ValueError:
            failed += 1
            continue
        se_td = math.sqrt(max(fit['V_jump'][1][1], 0.0))
        if se_td == 0.0 or (fit['tau_d'] / se_td) ** 2 < 10.0:
            weak_count += 1
        if abs(fit['tau_d']) >= 1e-9:
            ratios.append(fit['tau_y'] / fit['tau_d'])
    if not ratios:
        return {'ratios': [], 'weak_share': (weak_count + failed) / total}
    return {'ratios': ratios, 'weak_share': weak_count / total}

# ---------------------------------------------------------------------------
# 诊断：密度堆积、离散支撑、两侧构成
# ---------------------------------------------------------------------------

def _mccrary_discrete(records, cutoff):
    """McCrary 密度检验的离散评分适配：以每个评分点为箱，对箱计数取对数做局部线性。

    这是对评分点操纵/堆积的诊断，不能作为“政策随机化”的证明。
    """
    counts = {}
    for rec in records:
        counts[rec['score']] = counts.get(rec['score'], 0) + 1
    scores = sorted(counts)
    span_half = max(cutoff - min(scores), max(scores) - cutoff)
    h = min(IK_CONSTANTS[1] * span_half / 1.715 if span_half > 0 else 1.0, span_half)
    frame = [{'score': s, 'y': math.log(counts[s])} for s in scores]
    try:
        res = _single_jump(frame, max(h, 1.0), cutoff, 1, 'triangular')
        z = res['jump'] / res['se'] if res['se'] > 0 else None
        return {'log_density_jump': res['jump'], 'se': res['se'], 'z': z,
                'pilot_bandwidth': max(h, 1.0),
                'support_left': res['n_left'], 'support_right': res['n_right']}
    except ValueError:
        return {'log_density_jump': None, 'se': None, 'z': None,
                'pilot_bandwidth': max(h, 1.0),
                'note': '离散支撑点不足，密度检验不可估计'}

def _heaping(records, cutoff):
    counts = {}
    for rec in records:
        counts[rec['score']] = counts.get(rec['score'], 0) + 1
    scores = sorted(counts)
    n = len(records)
    flagged, max_ratio = [], 0.0
    for i, s in enumerate(scores):
        neigh = [counts[scores[j]] for j in (i - 1, i + 1) if 0 <= j < len(scores)]
        if not neigh:
            continue
        ratio = counts[s] / (sum(neigh) / len(neigh))
        max_ratio = max(max_ratio, ratio)
        if ratio >= 1.5 and counts[s] >= 3:
            flagged.append({'score': s, 'count': counts[s],
                            'mass_share': counts[s] / n, 'neighbor_ratio': ratio})
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    granularity = min((scores[i + 1] - scores[i] for i in range(len(scores) - 1)),
                      default=None)
    return {'heaped_points': flagged, 'max_neighbor_ratio': max_ratio,
            'top_mass_points': [{'score': s, 'mass_share': c / n} for s, c in top],
            'score_granularity': granularity,
            'unique_scores_total': len(scores)}

def _balance(records, frame_full, cutoff, bandwidth):
    """前定变量在门槛处的跳跃：只报告构成差异，不赋予随机化含义。"""
    out = {}
    for key in ('score_day', 'decision_day'):
        fr = [{'score': r['score'], 'y': r[key]} for r in
              sorted(records, key=lambda d: d['person'])]
        # 与主样本对齐（按 person 顺序一致即可，跳跃只依赖 score/y）
        try:
            res = _single_jump(fr, bandwidth, cutoff, 1, 'triangular')
            sd = math.sqrt(sum((x['y'] - sum(q['y'] for q in fr) / len(fr)) ** 2
                               for x in fr) / max(len(fr) - 1, 1))
            out[key] = {'jump': res['jump'], 'se': res['se'],
                        'standardized_diff': res['jump'] / sd if sd > 0 else None}
        except ValueError:
            out[key] = {'jump': None, 'se': None, 'standardized_diff': None}
    miss = [{'score': r['score'], 'y': 1.0 if r['y'] is None else 0.0}
            for r in frame_full]
    try:
        res = _single_jump(miss, bandwidth, cutoff, 1, 'triangular')
        out['outcome_missingness'] = {'jump': res['jump'], 'se': res['se']}
    except ValueError:
        out['outcome_missingness'] = {'jump': None, 'se': None}
    return out

# ---------------------------------------------------------------------------
# 规格封装与敏感性分析
# ---------------------------------------------------------------------------

def _estimate_cell(frame_y, frame_d, bandwidths, cutoff, degree, kernel, alpha,
                   boot_reps=0, seed=20260502):
    # 两个方程在同一批主体上联合：按各自带宽过滤后取主体交集并对齐行顺序，
    # 使第 i 行在两个方程中指向同一主体（跨方程协方差才有意义）。
    by_person_d = {r['person']: r for r in frame_d}
    fy, fd = [], []
    for ry in frame_y:
        if _kernel_weight(ry['score'] - cutoff, bandwidths[0], kernel) <= 0.0:
            continue
        rd = by_person_d.get(ry['person'])
        if rd is None or _kernel_weight(rd['score'] - cutoff, bandwidths[1], kernel) <= 0.0:
            continue
        fy.append(ry)
        fd.append(rd)
    fit = _joint_fit([fy, fd], bandwidths, cutoff, degree, kernel)
    Vj = fit['V_jump']
    boot = None
    if boot_reps:
        boot = _paired_bootstrap(fy, fd, bandwidths, cutoff,
                                 degree, kernel, boot_reps, seed)
    boot_ratios = boot['ratios'] if boot else []
    boot_weak = boot['weak_share'] if boot else None
    inference = _ratio_inference(fit['tau_y'], fit['tau_d'], Vj, alpha,
                                 boot_ratios, boot_weak)
    sup_y = _support(fy, bandwidths[0], cutoff)
    sup_d = _support(fd, bandwidths[1], cutoff)
    return {'inference': inference,
            'n_used': len(fy),
            'n_score_clusters': fit.get('n_clusters'),
            'variance_estimator': '按评分点聚类的联合三明治（Lee & Card 2008，G/(G-1) 校正）',
            'n_outcome_pool': len(frame_y), 'n_takeup_pool': len(frame_d),
            'support_outcome': {'left': sup_y[0], 'right': sup_y[1]},
            'support_takeup': {'left': sup_d[0], 'right': sup_d[1]},
            'bandwidths': {'outcome': bandwidths[0], 'takeup': bandwidths[1]}}

def _sensitivity(records, base_window, h_y, h_d, cutoff, degree, kernel, alpha):
    cells = []
    # 1) 带宽网格：估计目标恒为门槛依从者 LATE，外推范围与局部权重改变
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5):
        fy_full, _ = _window_frame(records, base_window)
        fy = _complete_case(fy_full)
        fd = [{'score': r['score'], 'y': float(r['received']), 'person': r['person']}
              for r in sorted(records, key=lambda d: d['person'])]
        try:
            cell = _estimate_cell(fy, fd, [h_y * mult, h_d * mult], cutoff,
                                  degree, kernel, alpha)
        except ValueError as exc:
            cells.append({'dimension': 'bandwidth', 'variant': f'{mult:g}x_IK',
                          'estimand_note': '同一门槛依从者 LATE；带宽改变局部权重与外推距离',
                          'error': str(exc)})
            continue
        cell['dimension'] = 'bandwidth'
        cell['variant'] = f'{mult:g}x_IK'
        cell['estimand_note'] = '同一门槛依从者 LATE；带宽改变局部权重与外推距离'
        cells.append(cell)
    # 2) 甜甜圈（按离散评分点剔除门槛两侧最邻近的 k 个评分点，而非固定距离；
    #    固定距离在粒度>1 的评分上可能一个点都剔不掉）。目标人群随之改变。
    unique_scores = sorted({r['score'] for r in records})
    left_points = sorted((s for s in unique_scores if s < cutoff), reverse=True)
    right_points = sorted(s for s in unique_scores if s >= cutoff)
    for k in (1, 2):
        excluded = set(left_points[:k]) | set(right_points[:k])
        if not excluded:
            cells.append({'dimension': 'donut', 'variant': f'exclude_{k}_nearest_each_side',
                          'estimand_note': '剔除门槛最邻近评分点后的局部依从者效应',
                          'error': '门槛两侧没有可剔除的评分点'})
            continue
        fy_full, _ = _window_frame(records, base_window)
        fy = [r for r in _complete_case(fy_full) if r['score'] not in excluded]
        fd = [{'score': r['score'], 'y': float(r['received']), 'person': r['person']}
              for r in sorted(records, key=lambda d: d['person'])
              if r['score'] not in excluded]
        try:
            cell = _estimate_cell(fy, fd, [h_y, h_d], cutoff, degree, kernel, alpha)
        except ValueError as exc:
            cell = {'dimension': 'donut',
                    'variant': f'exclude_{k}_nearest_each_side',
                    'estimand_note': '剔除门槛最邻近评分点后的局部依从者效应',
                    'excluded_scores': sorted(excluded), 'error': str(exc)}
            cells.append(cell)
            continue
        cell['dimension'] = 'donut'
        cell['variant'] = f'exclude_{k}_nearest_each_side'
        cell['estimand_note'] = '剔除门槛最邻近评分点后的局部依从者效应；估计目标为剩余局部依从者'
        cell['excluded_scores'] = sorted(excluded)
        cells.append(cell)
    # 3) 结果窗口：估计目标随随访窗口改变（不同 horizon 的效应）
    base_left, base_right = base_window
    window_grid = [(base_left, max(base_right // 2, 1)),
                   (base_left, base_right),
                   (base_left, base_right * 2)]
    for win in window_grid:
        fy_full, _ = _window_frame(records, win)
        fy = _complete_case(fy_full)
        fd = [{'score': r['score'], 'y': float(r['received']), 'person': r['person']}
              for r in sorted(records, key=lambda d: d['person'])]
        try:
            hy = _ik_bandwidth([r['score'] for r in fy], [r['y'] for r in fy],
                               cutoff, degree)
            cell = _estimate_cell(fy, fd, [hy, h_d], cutoff, degree, kernel, alpha)
        except ValueError as exc:
            cell = {'dimension': 'outcome_window', 'variant': f'[{win[0]},{win[1]})',
                    'estimand_note': '随访窗口改变，估计目标为该窗口经营结果的局部效应',
                    'error': str(exc)}
            cells.append(cell)
            continue
        cell['dimension'] = 'outcome_window'
        cell['variant'] = f'[{win[0]},{win[1]})'
        cell['estimand_note'] = '随访窗口改变，估计目标为该窗口经营结果的局部效应'
        cells.append(cell)
    return cells

# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def analyze_frd(snapshot, degree=1, kernel='triangular', alpha=0.05,
                boot_reps=399, seed=20260502, run_sensitivity=True):
    if degree not in (1, 2):
        raise ValueError('局部拟合阶数支持 1（线性）或 2（二次）')
    if kernel not in ('triangular', 'uniform'):
        raise ValueError('核支持 triangular 或 uniform')
    records, cutoff = build_persons(snapshot)
    window = tuple(snapshot.metadata['window'])
    baseline_summary = analyze(snapshot)

    frame_full, have = _window_frame(records, window)
    frame_y = _complete_case(frame_full)
    frame_d = [{'score': r['score'], 'y': float(r['received']),
                'person': r['person']}
               for r in sorted(records, key=lambda d: d['person'])]

    result = {
        'estimand': '门槛处依从者（因资格变化而改变领取行为者）的扶持领取局部效应 LATE = tau_y/tau_d',
        'cutoff': cutoff, 'running_variable': '干预前评分 score（单一门槛，不混用后补评分或其他门槛）',
        'eligibility_rule': 'Z = 1{score >= cutoff}，资格与实际领取 received 分开建模',
        'outcome_window': list(window), 'kernel': kernel, 'local_degree': degree,
        'baseline_summary': baseline_summary,
        'sample': {'applications': len(records),
                   'with_outcome_in_window': sum(have),
                   'without_outcome_in_window': len(records) - sum(have),
                   'eligible': sum(r['eligible'] for r in records),
                   'received': sum(r['received'] for r in records)},
    }

    # 带宽：结果方程与领取方程分别按 IK 选取，联合估计时各自携带
    try:
        h_y = _ik_bandwidth([r['score'] for r in frame_y],
                            [r['y'] for r in frame_y], cutoff, degree)
    except ValueError as exc:
        h_y = None
        result['bandwidth_warning'] = f'结果方程 IK 带宽不可用：{exc}'
    try:
        h_d = _ik_bandwidth([r['score'] for r in frame_d],
                            [r['y'] for r in frame_d], cutoff, degree)
    except ValueError as exc:
        h_d = None
        if 'bandwidth_warning' in result:
            result['bandwidth_warning'] += f'；领取方程 IK 带宽不可用：{exc}'
        else:
            result['bandwidth_warning'] = f'领取方程 IK 带宽不可用：{exc}'

    if h_y is None or h_d is None:
        result['primary'] = None
        result['identification'] = {'status': 'not_identified',
                                    'reason': result.get('bandwidth_warning', '带宽不可用')}
    else:
        try:
            cell = _estimate_cell(frame_y, frame_d, [h_y, h_d], cutoff,
                                  degree, kernel, alpha, boot_reps, seed)
        except ValueError as exc:
            result['primary'] = None
            result['identification'] = {'status': 'not_identified', 'reason': str(exc)}
        else:
            result['bandwidths_ik'] = {'outcome': h_y, 'takeup': h_d}
            result['primary'] = cell
            inf = cell['inference']
            ar = inf['anderson_rubin_95']
            if inf['weak_identification']:
                status = 'weakly_identified'
            elif ar['type'] == 'bounded':
                status = 'identified'
            elif ar['type'] == 'empty':
                status = 'model_rejected_empty_AR'
            else:
                status = 'unbounded_AR_set'
            result['identification'] = {
                'status': status,
                'rule': '领取跳跃 F<10 判弱识别；此时主推断为 AR 接受集合（可能有界但很宽、无界或整实线），'
                        '不得仅报告 Wald 比值的窄正态区间；AR 空集意味着联合模型被数据拒绝',
                'anderson_rubin_type': ar['type'],
                'ar_set_width': inf['ar_set_width'],
                'first_stage_F': inf['first_stage_F']}

    # 诊断（独立于主估计是否成功）
    density = _mccrary_discrete(records, cutoff)
    heaping = _heaping(records, cutoff)
    diag_bw = h_d if h_d is not None else \
        (max(r['score'] for r in records) - min(r['score'] for r in records)) / 2
    support_flag = _support_flag(records, cutoff, h_y, h_d, frame_y, frame_d)
    result['diagnostics'] = {
        'density_mccrary': density,
        'heaping_and_support': heaping,
        'composition_balance': _balance(records, frame_full, cutoff, diag_bw),
        'discrete_support_flag': support_flag,
        'interpretation': [
            '密度与构成检查是评分操纵/堆积/选择性流失的诊断，不构成政策随机化或外生性的证明。',
            '离散评分下有效支撑点不足时，局部多项式结果不可与连续运行变量情形等同解读。'],
    }
    # 支撑稀疏作为识别可靠性的一级警告，与主估计状态并列呈现
    sparse_equations = [e['equation'] for e in support_flag['equations']
                        if e.get('sparse')]
    if result['primary'] is not None:
        result['identification']['discrete_support_sparse'] = bool(sparse_equations)
        result['identification']['sparse_equations'] = sparse_equations
        if sparse_equations:
            result['identification']['support_warning'] = \
                '带宽内门槛一侧唯一评分点不足 5 个，局部拟合自由度紧张；' \
                '结果应结合敏感性与 AR 集合审慎解读，不能视为连续评分下的标准断点估计'

    if run_sensitivity and h_y is not None and h_d is not None:
        result['sensitivity'] = _sensitivity(records, window, h_y, h_d,
                                             cutoff, degree, kernel, alpha)
    else:
        result['sensitivity'] = []
    return result

def _support_flag(records, cutoff, h_y, h_d, frame_y, frame_d):
    scores = sorted({r['score'] for r in records})
    granularity = min((scores[i + 1] - scores[i] for i in range(len(scores) - 1)),
                      default=None)
    flags = []
    for name, frame, bw in (('outcome', frame_y, h_y), ('takeup', frame_d, h_d)):
        if bw is None:
            flags.append({'equation': name, 'available': False})
            continue
        left, right = _support(frame, bw, cutoff)
        flags.append({'equation': name, 'bandwidth': bw,
                      'unique_scores_left': left, 'unique_scores_right': right,
                      'sparse': min(left, right) < 5,
                      'threshold_rule': '任一侧唯一评分点 < 5 判定为离散支撑稀疏'})
    return {'score_granularity': granularity, 'equations': flags}

def _json_clean(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    return obj

def inspect_frd(payload, **kwargs):
    snapshot = load(payload, SCHEMAS)
    report = analyze_frd(snapshot, **kwargs)
    return {'schema_version': 1, 'input_digest': snapshot.digest,
            'data_version': snapshot.metadata.get('data_version'),
            'frd': _json_clean(report)}

def main(argv=None):
    parser = argparse.ArgumentParser(description='模糊断点局部效应与识别诊断')
    parser.add_argument('path', help='资料 JSON 路径')
    parser.add_argument('--degree', type=int, default=1, choices=(1, 2),
                        help='局部多项式阶数（默认 1）')
    parser.add_argument('--kernel', default='triangular',
                        choices=('triangular', 'uniform'))
    parser.add_argument('--boot-reps', type=int, default=399,
                        help='配对分层 bootstrap 次数，0 关闭')
    parser.add_argument('--no-sensitivity', action='store_true',
                        help='关闭带宽/甜甜圈/窗口敏感性分析')
    parser.add_argument('--seed', type=int, default=20260502)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.path).read_text(encoding='utf-8'))
    report = inspect_frd(payload, degree=args.degree, kernel=args.kernel,
                         boot_reps=args.boot_reps, seed=args.seed,
                         run_sensitivity=not args.no_sensitivity)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))

if __name__ == '__main__':
    main()
