"""FRD 数值管线验证：用已知生成过程的合成数据检验恢复能力与弱识别行为。

合成数据为人工生成，不对应真实主体；脚本独立于 app 包的业务入口。
运行：python -I tools/sim_check.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.frd import inspect_frd


def simulate(n=600, cutoff=60, tau=8.0, complier_share=0.5, weak=False, seed=7):
    """标准 FRD 潜在依从类型 DGP，使真实 LATE 恰好等于 tau。

    主体类型：始终领取者（份额 p_a）、从不领取者、依从者（份额 π_c）。
    D(0)=始终领取指示；D(1)=D(0) 或 依从者。故领取跳跃 E[D|1]-E[D|0]=π_c。
    Y = y0 + tau * C * D：效应只作用于实际领取的依从者，
    因而 ITT_Y = tau*pi_c，Wald = tau，与 tau 本身相等。
    weak=True 时 pi_c≈0.06，制造弱识别。
    """
    rng = random.Random(seed)
    pi_c = 0.06 if weak else complier_share
    p_a = 0.15
    apps, outs = [], []
    for i in range(n):
        score = rng.randint(40, 80)
        z = 1 if score >= cutoff else 0
        complier = rng.random() < pi_c
        always = (not complier) and rng.random() < p_a
        d0 = 1 if always else 0
        received = 1 if (d0 or (complier and z)) else 0
        decision_day = 5
        y0 = 20 + 0.3 * (score - cutoff) + rng.gauss(0, 4.0)
        value = y0 + tau * complier * received
        pid = f'p{i:04d}'
        apps.append({'id': f'a{i:04d}', 'revision': 1, 'person': pid,
                     'score': score, 'score_day': 0, 'decision_day': decision_day,
                     'received': bool(received)})
        outs.append({'id': f'y{i:04d}', 'revision': 1, 'person': pid,
                     'day': 10, 'value': round(value, 3)})
        # 多期结果：窗口内取均值后仍应反映同一效应
        outs.append({'id': f'y{i:04d}b', 'revision': 1, 'person': pid,
                     'day': 30, 'value': round(value + rng.gauss(0, 1.0), 3)})
        # 更晚的结果点：半宽窗口 [0,45) 不含它，用于检验窗口敏感性确实改变样本
        outs.append({'id': f'y{i:04d}c', 'revision': 1, 'person': pid,
                     'day': 60, 'value': round(value + rng.gauss(0, 1.5), 3)})
    return {'schema_version': 1,
            'metadata': {'data_version': 'sim-strong' if not weak else 'sim-weak',
                         'cutoff': cutoff, 'window': [0, 90]},
            'collections': {'applications': apps, 'outcomes': outs}}


def summarize(tag, payload, expect_weak=False):
    report = inspect_frd(payload, boot_reps=199)['frd']
    pri = report['primary']
    ident = report['identification']
    print(f'== {tag} ==')
    print('  identification:', ident['status'], '| F =',
          None if pri is None else round(pri['inference']['first_stage_F'], 2))
    if pri is not None:
        inf = pri['inference']
        print('  tau_d =', round(inf['tau_d'], 3), ' tau_y =', round(inf['tau_y'], 3))
        wr = inf['wald_ratio_point_estimate']
        wse = inf['wald_ratio_delta_se_reference']
        print('  wald 比值(参照) =', None if wr is None else round(wr, 3),
              ' delta-se(参照) =', None if wse is None else round(wse, 3))
        ar = inf['anderson_rubin_95']
        print('  AR95 集合类型 =', ar['type'], ' bounds =', ar['bounds'])
        print('  AR 宽度 =', None if inf['ar_set_width'] is None
              else round(inf['ar_set_width'], 3))
        print('  boot p95 =', inf['bootstrap_p95'],
              ' 弱F份额 =', inf['bootstrap_weak_F_share'])
        print('  bandwidths:', {k: round(v, 2) for k, v in pri['bandwidths'].items()})
        print('  support:', pri['support_outcome'], pri['support_takeup'],
              ' n_used =', pri['n_used'])
        if expect_weak:
            assert inf['weak_identification'] and ident['status'] == 'weakly_identified'
            # 弱识别的可靠标志：F<10 被标记，且聚类 bootstrap 复现样本多数仍 F<10；
            # AR 集合宽度受少簇保守性影响，跨样本宽度比较不是稳定判据，故不采用。
            assert inf['bootstrap_weak_F_share'] > 0.6, '弱一阶段复现样本应多数 F<10'
        else:
            assert ar['type'] == 'bounded', '强一阶段 AR 应为有界区间'
            assert ar['bounds'][0] <= 8.0 <= ar['bounds'][1], \
                f"AR 区间应覆盖真实 LATE=8，实际 {ar['bounds']}"
    dens = report['diagnostics']['density_mccrary']
    print('  mccrary z =', None if dens['z'] is None else round(dens['z'], 2))
    print('  sensitivity cells:', len(report['sensitivity']))
    print()
    return report


def main():
    strong = simulate(weak=False)
    weak = simulate(weak=True)
    r1 = summarize('强一阶段（真 LATE=8）', strong)
    summarize('弱一阶段（弱识别）', weak, expect_weak=True)

    cells = r1['sensitivity']
    assert {c['dimension'] for c in cells} == {'bandwidth', 'donut', 'outcome_window'}
    for c in cells:
        assert 'estimand_note' in c and 'variant' in c
    # 实际使用样本数随带宽收缩而下降
    bw_cells = [c for c in cells if c['dimension'] == 'bandwidth' and 'error' not in c]
    ns = [c['n_used'] for c in bw_cells]
    assert ns == sorted(ns), '带宽扩大时使用样本数应单调不减'

    q = inspect_frd(strong, degree=2, boot_reps=99)['frd']
    assert q['primary'] is not None and q['local_degree'] == 2
    print('二次规格 AR:', q['primary']['inference']['anderson_rubin_95']['bounds'])

    # JSON 可序列化且不含 NaN/Infinity
    blob = json.dumps(inspect_frd(strong), allow_nan=False)
    assert 'NaN' not in blob and 'Infinity' not in blob
    print('全部断言通过。')


if __name__ == '__main__':
    main()
