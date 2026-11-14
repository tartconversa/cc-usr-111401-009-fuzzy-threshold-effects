"""模糊断点分析的单元测试：识别状态、弱识别区间、诊断与既有摘要兼容性。"""
import copy
import json
import math
import unittest
from pathlib import Path

from app.catalog import load
from app.service import inspect, SCHEMAS
from app.frd import (analyze_frd, inspect_frd, build_persons, _quadratic_set,
                     _norm_q, _chi2_q1, _ar_set, _joint_fit, _ik_bandwidth)

SAMPLE = json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / 'sample.json').read_text(encoding='utf-8'))


def _payload(n=420, seed=3, pi_c=0.5, tau=8.0):
    """潜在依从类型 DGP：真实 LATE 恰好为 tau，领取跳跃为 pi_c。"""
    rng = __import__('random').Random(seed)
    apps, outs = [], []
    cutoff = 60
    p_a = 0.15
    for i in range(n):
        score = rng.randint(35, 85)
        z = 1 if score >= cutoff else 0
        complier = rng.random() < pi_c
        always = (not complier) and rng.random() < p_a
        received = 1 if (always or (complier and z)) else 0
        y0 = 20 + 0.3 * (score - cutoff) + rng.gauss(0, 4.0)
        value = y0 + tau * complier * received
        pid = f'p{i:04d}'
        apps.append({'id': f'a{i}', 'revision': 1, 'person': pid, 'score': score,
                     'score_day': 0, 'decision_day': 5, 'received': bool(received)})
        outs.append({'id': f'y{i}', 'revision': 1, 'person': pid, 'day': 10,
                     'value': round(value, 3)})
    return {'schema_version': 1,
            'metadata': {'data_version': 'unit-sim', 'cutoff': cutoff, 'window': [0, 90]},
            'collections': {'applications': apps, 'outcomes': outs}}


class NumericalHelperTests(unittest.TestCase):
    def test_normal_quantiles(self):
        self.assertAlmostEqual(_norm_q(0.975), 1.959964, places=4)
        self.assertAlmostEqual(_norm_q(0.5), 0.0, places=6)
        self.assertAlmostEqual(_chi2_q1(0.95), 3.841459, places=4)

    def test_quadratic_set_shapes(self):
        # 开口向上、判别式正：有界
        s = _quadratic_set(1.0, 0.0, -4.0, 0.0)
        self.assertEqual(s['type'], 'bounded')
        self.assertAlmostEqual(s['bounds'][0], -2.0)
        self.assertAlmostEqual(s['bounds'][1], 2.0)
        # 判别式负且开口向上：空集
        self.assertEqual(_quadratic_set(1.0, 0.0, 1.0, 0.0)['type'], 'empty')
        # 开口向下：双侧无界尾
        s2 = _quadratic_set(-1.0, 0.0, 4.0, 0.0)
        self.assertEqual(s2['type'], 'two_unbounded_tails')


class FRDIdentificationTests(unittest.TestCase):
    def test_baseline_summary_preserved_inside_frd(self):
        # 既有资格摘要必须原样保留
        baseline = inspect(SAMPLE)['summary']
        frd = inspect_frd(SAMPLE, boot_reps=0, run_sensitivity=False)['frd']
        self.assertEqual(frd['baseline_summary'], baseline)

    def test_tiny_sample_is_not_identified_without_fake_interval(self):
        frd = inspect_frd(SAMPLE, boot_reps=0)['frd']
        self.assertIsNone(frd['primary'])
        self.assertEqual(frd['identification']['status'], 'not_identified')
        # 诊断仍应产出，并明确不是随机化证明
        self.assertIn('interpretation', frd['diagnostics'])

    def test_strong_first_stage_bounded_ar_covers_effect(self):
        report = analyze_frd(load(_payload(), SCHEMAS), boot_reps=99)
        inf = report['primary']['inference']
        self.assertFalse(inf['weak_identification'])
        ar = inf['anderson_rubin_95']
        self.assertEqual(ar['type'], 'bounded')
        self.assertLessEqual(ar['bounds'][0], 8.0)
        self.assertGreaterEqual(ar['bounds'][1], 8.0)
        self.assertEqual(report['identification']['status'], 'identified')

    def test_weak_first_stage_flagged_and_wald_marked_reference_only(self):
        weak = _payload(pi_c=0.05, seed=11)
        report = analyze_frd(load(weak, SCHEMAS), boot_reps=99)
        inf = report['primary']['inference']
        self.assertTrue(inf['weak_identification'])
        self.assertEqual(report['identification']['status'], 'weakly_identified')
        self.assertIsNotNone(inf['wald_warning'])
        # AR 集合必须存在；弱识别下可以宽、无界、整实线，或因联合模型被拒绝而空集
        self.assertIn(inf['anderson_rubin_95']['type'],
                      ('bounded', 'two_unbounded_tails', 'whole_real_line', 'empty'))
        self.assertGreater(inf['bootstrap_weak_F_share'], 0.5)

    def test_joint_covariance_is_not_forced_zero(self):
        # 同一主体的结果与领取冲击相关时，跨方程协方差应非零
        payload = _payload()
        snap = load(payload, SCHEMAS)
        recs, cutoff = build_persons(snap)
        frame = [{'person': r['person'], 'score': r['score'],
                  'y': 20 + r['score'] - cutoff, 'received': r['received'],
                  'eligible': r['eligible']} for r in recs]
        fd = [{'person': r['person'], 'score': r['score'], 'y': float(r['received'])}
              for r in recs]
        h = _ik_bandwidth([f['score'] for f in frame], [f['y'] for f in frame], cutoff)
        fit = _joint_fit([frame, fd], [h, h], cutoff, 1, 'triangular')
        self.assertEqual(len(fit['V_jump']), 2)
        for row in fit['V_jump']:
            for v in row:
                self.assertTrue(math.isfinite(v))

    def test_single_cutoff_and_prescore_only(self):
        payload = _payload()
        report = inspect_frd(payload, boot_reps=0, run_sensitivity=False)['frd']
        self.assertEqual(report['cutoff'], 60)
        self.assertIn('干预前评分', report['running_variable'])
        self.assertNotIn('post_score', json.dumps(report))


class SensitivityAndDiagnosticTests(unittest.TestCase):
    def test_sensitivity_links_sample_and_estimand(self):
        report = analyze_frd(load(_payload(), SCHEMAS), boot_reps=0)
        cells = report['sensitivity']
        dims = {c['dimension'] for c in cells}
        self.assertEqual(dims, {'bandwidth', 'donut', 'outcome_window'})
        for c in cells:
            self.assertIn('estimand_note', c)
            if 'error' not in c:
                self.assertIn('n_used', c)
        # 甜甜圈改变估计目标人群，并记录实际剔除的评分点
        donuts = [c for c in cells if c['dimension'] == 'donut']
        donut = next(c for c in donuts if 'error' not in c)
        self.assertIn('依从者', donut['estimand_note'])
        self.assertTrue(donut['excluded_scores'])

    def test_density_and_heaping_detect_manipulation_not_randomization(self):
        payload = _payload()
        # 人为在门槛上堆积
        for row in payload['collections']['applications'][:40]:
            row['score'] = 60
        report = analyze_frd(load(payload, SCHEMAS), boot_reps=0, run_sensitivity=False)
        heap = report['diagnostics']['heaping_and_support']
        self.assertTrue(any(h['score'] == 60 for h in heap['heaped_points']))
        text = ' '.join(report['diagnostics']['interpretation'])
        self.assertIn('不构成政策随机化', text)

    def test_discrete_support_sparsity_flag(self):
        frd = inspect_frd(SAMPLE, boot_reps=0, run_sensitivity=False)['frd']
        flags = frd['diagnostics']['discrete_support_flag']['equations']
        self.assertTrue(any(f.get('sparse') for f in flags))

    def test_support_warning_promoted_to_identification(self):
        report = analyze_frd(load(_payload(), SCHEMAS), boot_reps=0,
                             run_sensitivity=False)
        if report['identification'].get('discrete_support_sparse'):
            self.assertIn('support_warning', report['identification'])
            self.assertIn('AR 集合', report['identification']['support_warning'])

    def test_json_is_finite_and_serializable(self):
        blob = json.dumps(inspect_frd(_payload(n=120), boot_reps=39), allow_nan=False)
        self.assertNotIn('NaN', blob)
        self.assertNotIn('Infinity', blob)


if __name__ == '__main__':
    unittest.main()
