"""Regenerate every analysis chart from fixed data. One PNG + one CSV per chart.

Usage: python make_charts.py <asset-dir>
Each chart writes 차트-NN.png and 차트-NN.csv (the exact rows plotted), so the figure can be rebuilt
and audited without rerunning any model.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

PREP = Path('F:/rnd_github/pcnrnd/Population_decline/platform/data/prepared')
EVID = Path('F:/etc/review-system/artifacts/active/operations/20260916-population-completion')
ASSET = Path(sys.argv[1])
ASSET.mkdir(parents=True, exist_ok=True)

for cand in ('Malgun Gothic', 'NanumGothic', 'Gulim', 'Batang'):
    if any(f.name == cand for f in font_manager.fontManager.ttflist):
        rcParams['font.family'] = cand
        break
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 200
rcParams['savefig.bbox'] = 'tight'
PALETTE = ['#2F5D8C', '#C0504D', '#4F8A5B', '#8064A2', '#D18A2B']


def read(rel):
    with (PREP / rel).open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def num(v):
    v = (v or '').replace(',', '').strip()
    try:
        return float(v)
    except ValueError:
        return None


def dump(name, header, rows):
    with (ASSET / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save(fig, name):
    fig.savefig(ASSET / (name + '.png'))
    plt.close(fig)
    print('wrote', name)


def ym_label(ym):
    return '%d-%02d' % (ym // 100, ym % 100)


# ---------- 차트-01 기간 커버리지 ----------
fams = [
    ('남원 KT 방문객\n(월 × 행정동 23)', 201901, 202310, PALETTE[0]),
    ('남원 BC카드 소비\n(월 × 행정동 23)', 201901, 202312, PALETTE[1]),
    ('강원 GWTO 관광지표\n(월 × 시군구 18)', 202001, 202512, PALETTE[2]),
    ('서울 노원 축제\n(일 × 지점 1)', 202508, 202509, PALETTE[3]),
]
gap = (202001, 202112)
fig, ax = plt.subplots(figsize=(10, 3.6))
rows = []
for i, (label, lo, hi, color) in enumerate(fams):
    x0 = lo // 100 * 12 + lo % 100
    x1 = hi // 100 * 12 + hi % 100
    ax.barh(i, x1 - x0 + 1, left=x0, height=0.5, color=color)
    rows.append([label.replace('\n', ' '), lo, hi])
    if '소비' in label:
        g0 = gap[0] // 100 * 12 + gap[0] % 100
        g1 = gap[1] // 100 * 12 + gap[1] % 100
        ax.barh(i, g1 - g0 + 1, left=g0, height=0.5, color='white', hatch='///',
                edgecolor=color, linewidth=0.8)
ticks = [y * 12 + 1 for y in range(2019, 2026)]
ax.set_yticks(range(len(fams)))
ax.set_yticklabels([f[0] for f in fams], fontsize=8)
ax.set_xticks(ticks)
ax.set_xticklabels([str(y) for y in range(2019, 2026)], fontsize=8)
ax.set_xlabel('연도', fontsize=9)
ax.invert_yaxis()
ax.grid(axis='x', alpha=.25)
ax.set_title('데이터셋별 관측 기간과 관측 단위 (빗금 = 원천 공백 2020–2021)', fontsize=10)
dump('차트-01', ['dataset', 'start_ym', 'end_ym'], rows)
save(fig, '차트-01')

# ---------- 차트-02 남원 방문객 월별 ----------
kt = read('namwon_kt_visitors/01_monthly_dong_visitors.csv')
by_m = defaultdict(float)
for r in kt:
    by_m[int(r['base_ym'])] += num(r['nonlocal_visitors']) or 0
ms = sorted(by_m)
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot([ym_label(m) for m in ms], [by_m[m] / 1e4 for m in ms], color=PALETTE[0], lw=1.6)
ax.set_ylabel('외지인 방문객 (만 명)', fontsize=9)
ax.set_xticks(range(0, len(ms), 6))
ax.set_xticklabels([ym_label(ms[i]) for i in range(0, len(ms), 6)], rotation=45, ha='right', fontsize=8)
ax.grid(alpha=.25)
ax.set_title('남원시 월별 외지인 방문객 합계 (23개 행정동 합, 2019-01~2023-10)', fontsize=10)
dump('차트-02', ['base_ym', 'nonlocal_visitors_sum'], [[m, by_m[m]] for m in ms])
save(fig, '차트-02')

# ---------- 차트-03 남원 소비 월별 (공백 가시화) ----------
bc = read('namwon_bccard_consumption/02_dong_industry_sales.csv')
by_s = defaultdict(float)
for r in bc:
    by_s[int(r['base_ym'])] += num(r['sales_est_krw']) or 0
sm = sorted(by_s)
full = []
y, m = 201901, None
cur = 201901
while cur <= 202312:
    full.append(cur)
    yy, mm = divmod(cur, 100)
    mm += 1
    if mm == 13:
        yy, mm = yy + 1, 1
    cur = yy * 100 + mm
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot([ym_label(m) for m in full], [by_s.get(m, float('nan')) / 1e8 for m in full],
        color=PALETTE[1], lw=1.6, marker='o', ms=2.5)
ax.axvspan(full.index(202001), full.index(202112), color='#999999', alpha=.16)
ax.text(full.index(202001) + 6, ax.get_ylim()[1] * .92, '원천 공백 24개월\n(보간하지 않음)',
        fontsize=8, ha='left', va='top', color='#444444')
ax.set_ylabel('추정 매출 (억 원)', fontsize=9)
ax.set_xticks(range(0, len(full), 6))
ax.set_xticklabels([ym_label(full[i]) for i in range(0, len(full), 6)], rotation=45, ha='right', fontsize=8)
ax.grid(alpha=.25)
ax.set_title('남원시 월별 BC카드 추정 매출 합계 (2019-01~2023-12, 2020–2021 원천 부재)', fontsize=10)
dump('차트-03', ['base_ym', 'sales_est_krw_sum_or_blank'], [[m, by_s.get(m, '')] for m in full])
save(fig, '차트-03')

# ---------- 차트-04 남원 지역 편중 ----------
v_d = defaultdict(float)
for r in kt:
    v_d[r['dong_code']] += num(r['nonlocal_visitors']) or 0
s_d = defaultdict(float)
for r in bc:
    s_d[r['dong_name'].strip()] += num(r['sales_est_krw']) or 0
mp = json.loads(Path('F:/rnd_github/pcnrnd/Population_decline/platform/backend/xops-service/src/realdata/'
                     'namwon_dong_map.json').read_text(encoding='utf-8'))
code2name = {e['dong_code']: e['dong_name'] for e in mp['entries']}
vt, st = sum(v_d.values()), sum(s_d.values())
order = sorted(v_d, key=lambda c: -v_d[c])
names = [code2name.get(c, c) for c in order]
fig, ax = plt.subplots(figsize=(10, 3.8))
x = range(len(order))
ax.bar([i - .2 for i in x], [100 * v_d[c] / vt for c in order], width=.4, color=PALETTE[0], label='방문객 점유율 %')
ax.bar([i + .2 for i in x], [100 * s_d.get(code2name.get(c, ''), 0) / st for c in order], width=.4,
       color=PALETTE[1], label='소비 점유율 %')
ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
ax.set_ylabel('전체 대비 점유율 (%)', fontsize=9)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=.25)
ax.set_title('남원 23개 행정동의 점유율 편중 — 각 지표를 자기 합계로 정규화한 값 (단위가 달라 절대량 비교 아님)',
             fontsize=9.5)
dump('차트-04', ['dong_name', 'visitor_share_pct', 'sales_share_pct'],
     [[code2name.get(c, c), 100 * v_d[c] / vt, 100 * s_d.get(code2name.get(c, ''), 0) / st] for c in order])
save(fig, '차트-04')

# ---------- 차트-05 GWTO 테이블별 유효구간 ----------
q = json.loads((EVID / 'stage6-full-data' / 'quality.json').read_text(encoding='utf-8'))
gw = q['families']['gwto_kt_tourism_indicators']['per_table']
items = [(k, v['period']['min'], v['period']['max']) for k, v in gw.items() if v.get('period')]
items.sort(key=lambda t: (int(t[1]), int(t[2])))
fig, ax = plt.subplots(figsize=(10, 5.2))
for i, (k, lo, hi) in enumerate(items):
    lo, hi = int(lo), int(hi)
    x0 = lo // 100 * 12 + lo % 100
    x1 = hi // 100 * 12 + hi % 100
    ax.barh(i, x1 - x0 + 1, left=x0, height=0.6, color=PALETTE[2])
ax.set_yticks(range(len(items)))
ax.set_yticklabels([k.replace('.csv', '') for k, _, _ in items], fontsize=6.5)
ax.set_xticks([y * 12 + 1 for y in range(2020, 2026)])
ax.set_xticklabels([str(y) for y in range(2020, 2026)], fontsize=8)
ax.invert_yaxis()
ax.grid(axis='x', alpha=.25)
ax.set_title('강원 GWTO 지표 테이블별 유효 구간 — 시작월이 제각각이라 분모가 다르다', fontsize=10)
dump('차트-05', ['table', 'start_ym', 'end_ym'], [[k, lo, hi] for k, lo, hi in items])
save(fig, '차트-05')

# ---------- 차트-06 노원 축제 일별 ----------
nw = read('nowon_kt_festival/02_daily_visitors.csv')
col = next((c for c in nw[0] if 'visitor' in c.lower() and num(nw[0][c]) is not None), None)
ds = [r['date'] for r in nw]
vs = [num(r[col]) or 0 for r in nw]
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(ds, vs, color=PALETTE[3])
ax.set_xticks(range(0, len(ds), 3))
ax.set_xticklabels([ds[i] for i in range(0, len(ds), 3)], rotation=45, ha='right', fontsize=8)
ax.set_ylabel(col, fontsize=9)
ax.grid(axis='y', alpha=.25)
ax.set_title('서울 노원 축제 일별 방문객 (%s ~ %s, 단일 지점)' % (ds[0], ds[-1]), fontsize=10)
dump('차트-06', ['date', col], [[d, v] for d, v in zip(ds, vs)])
save(fig, '차트-06')

# ---------- 차트-07 모델 롤링 결과 (기존 확정 증거 재사용) ----------
grid = json.loads((EVID / 'stage4-model' / 'grid.json').read_text(encoding='utf-8'))['grid']
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, (target, title) in zip(axes, [('nonlocal_visitors', '남원 방문객'), ('observed_sales_krw', '남원 소비')]):
    for name, color, lab in [('A0_shipped_zero', '#999999', '개선 전'), ('C2_yoy_prev_lad5', PALETTE[0], '개선 후(v0.3)')]:
        col = grid[target][name]
        cuts = sorted(col, key=int)
        ax.plot(range(len(cuts)), [col[c]['ratio'] for c in cuts], color=color, lw=1.5, label=lab)
    ax.axhline(1.0, color=PALETTE[1], ls='--', lw=1.2)
    ax.text(0.5, 1.02, '기준선 = 1.0 (아래면 승급 가능)', fontsize=7.5, color=PALETTE[1])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('평가 구간 (시간순)', fontsize=8.5)
    ax.set_ylabel('MAE 비 (모델 / 기준선)', fontsize=8.5)
    ax.set_ylim(0.4, 2.2)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
rows = []
for target in ('nonlocal_visitors', 'observed_sales_krw'):
    for name in ('A0_shipped_zero', 'C2_yoy_prev_lad5'):
        for c in sorted(grid[target][name], key=int):
            rows.append([target, name, c, grid[target][name][c]['ratio']])
dump('차트-07', ['target', 'variant', 'cutoff', 'mae_ratio'], rows)
fig.suptitle('구간별 MAE 비 — 확정된 기존 증거 재사용 (재학습 없음)', fontsize=10.5)
save(fig, '차트-07')
print('all charts written to', ASSET)
