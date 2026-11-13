"""现有业务摘要，不推断未观察到的业务关系。"""
from statistics import mean

def analyze(snapshot):
    results = {'below': {'n': 0, 'received': 0, 'outcomes': []}, 'above': {'n': 0, 'received': 0, 'outcomes': []}}
    applications = snapshot.rows('applications')
    if len({a['person'] for a in applications}) != len(applications):
        raise ValueError('同主体存在多个申请，需选择资格轮次')
    left, right = snapshot.metadata['window']
    for row in applications:
        if row['score_day'] > row['decision_day']:
            raise ValueError('评分发生在资格决定之后')
        side = 'above' if row['score'] >= snapshot.metadata['cutoff'] else 'below'
        group = results[side]
        group['n'] += 1
        group['received'] += int(row['received'])
        ys = [o['value'] for o in snapshot.rows('outcomes') if o['person'] == row['person']
              and left <= o['day'] - row['decision_day'] < right]
        if ys:
            group['outcomes'].append(mean(ys))
    return {k: {'n': v['n'], 'received': v['received'], 'outcome_mean': mean(v['outcomes']) if v['outcomes'] else None}
            for k, v in results.items()}
