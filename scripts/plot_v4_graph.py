"""Publication figure of the v4 LangGraph review chain (src/medical_jury/v4/graph.py, review-v4-graph-3).

Double-column width (183 mm), Arial-metric font (Liberation Sans; set SANS = 'Arial' where installed), Okabe-Ito colour-blind-safe palette,
fonts embedded as TrueType. Writes PDF, SVG and a 600 dpi PNG to docs/figures/.
Usage: python scripts/plot_v4_graph.py   (needs matplotlib; no project imports, no network)"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'figures'
W, H = 183.0, 124.0                                   # mm
SANS, MONO = 'Liberation Sans', 'Liberation Mono'
INK, GREY, LIGHT = '#1a1a1a', '#5f5f5f', '#9a9a9a'
LLM_E, LLM_F = '#0072B2', '#E3EEF7'                   # Okabe-Ito blue
HUM_E, HUM_F = '#D55E00', '#FBE6D9'                   # Okabe-Ito vermillion
CODE_E, CODE_F = '#4d4d4d', '#FFFFFF'
LW = 0.6

mpl.rcParams.update({'font.family': SANS, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'font.size': 6, 'axes.linewidth': LW})

fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')


def node(cx, cy, w, h, ident, lines, kind='code', dashed=False):
    edge, face = {'llm': (LLM_E, LLM_F), 'human': (HUM_E, HUM_F), 'code': (CODE_E, CODE_F)}[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                lw=LW * (1.4 if kind != 'code' else 1), ec=edge, fc=face,
                                ls=(0, (2.2, 1.4)) if dashed else '-', zorder=3))
    top = cy + h / 2 - 3.3
    ax.text(cx, top, ident, family=MONO, weight='bold', fontsize=5.7, color=INK, ha='center', va='center', zorder=4)
    for i, line in enumerate(lines):
        ax.text(cx, top - 3.4 - i * 2.6, line, fontsize=5.0, color=GREY, ha='center', va='center', zorder=4)
    return {'l': cx - w / 2, 'r': cx + w / 2, 't': cy + h / 2, 'b': cy - h / 2, 'x': cx, 'y': cy}


def arrow(points, dashed=False, color=INK):
    style = (0, (2.6, 1.6)) if dashed else '-'
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=LW, ls=style, solid_capstyle='butt', zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>,head_length=1.9,head_width=1.0',
                                 mutation_scale=1, lw=LW, color=color, ls=style, shrinkA=0, shrinkB=0.2, zorder=2))


def label(x, y, text, ha='center', va='center', italic=False, color=GREY, size=5.0, mono=False):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, style='italic' if italic else 'normal',
            family=MONO if mono else SANS, zorder=5,
            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none'))


# ---------------------------------------------------------------- row A (y = 88): screen, route, extract, fan-out, check
YA, YB, YH = 88.0, 46.0, 21.0
ax.add_patch(plt.Circle((4.0, YA), 1.7, color=INK, zorder=3))
ax.text(4.0, YA - 4.0, 'START', fontsize=5.0, color=GREY, ha='center', family=MONO)

scr = node(17.0, YA, 17, 14, 'screen_input', ['input screen', '(v4.2-dev)'], dashed=True)
arrow([(5.7, YA), (scr['l'], YA)])

dx = 34.1                                             # route_arm decision
ax.add_patch(Polygon([(dx - 4.6, YA), (dx, YA + 4.6), (dx + 4.6, YA), (dx, YA - 4.6)], closed=True, lw=LW,
                     ec=CODE_E, fc='white', zorder=3))
ax.text(dx, YA, 'arm', fontsize=5.4, color=INK, ha='center', va='center', zorder=4)
ax.text(dx, YA + 6.3, 'route_arm', family=MONO, fontsize=5.0, color=GREY, ha='center')
arrow([(scr['r'], YA), (dx - 4.6, YA)])

ext = node(53.6, YA, 21, 14, 'extract', ['LLM call 1', 'facts, medicines, quotes'], 'llm')
arrow([(dx + 4.6, YA), (ext['l'], YA)], dashed=True)
label((dx + 4.6 + ext['l']) / 2, YA + 2.4, 'full', italic=True)

FORK, JOIN = 68.5, 99.5
TRIO = (103.0, YA, 73.0)
for x in (FORK, JOIN):
    ax.add_patch(Rectangle((x - 0.6, TRIO[2] - 6.0), 1.2, TRIO[0] - TRIO[2] + 12.0, color=INK, zorder=3))
arrow([(ext['r'], YA), (FORK - 0.6, YA)], dashed=True)
label((ext['r'] + FORK) / 2, YA + 2.4, 'ok', italic=True)
ax.text(FORK, TRIO[0] + 7.8, 'fork', fontsize=5.0, color=GREY, ha='center', style='italic')
ax.text(JOIN, TRIO[0] + 7.8, 'join', fontsize=5.0, color=GREY, ha='center', style='italic')
ax.text((FORK + JOIN) / 2, TRIO[0] + 7.8, 'one parallel superstep', fontsize=5.0, color=GREY, ha='center',
        style='italic')

ver = node(84, TRIO[0], 23, 11.5, 'verify_extraction', ['quotes, drug coverage'])
rul = node(84, TRIO[1], 23, 11.5, 'rule_layer', ['10 rules (declarative)'])
sel = node(84, TRIO[2], 23, 11.5, 'select_criteria', ['general + disease items'])
for n in (ver, rul, sel):
    arrow([(FORK + 0.6, n['y']), (n['l'], n['y'])])
    arrow([(n['r'], n['y']), (JOIN - 0.6, n['y'])])

chk = node(114, YA, 21, 14, 'check', ['LLM call 2', 'criteria + open review'], 'llm')
arrow([(JOIN + 0.6, YA), (chk['l'], YA)])
agg = node(138.5, YA, 20, 14, 'aggregate', ['merge + reconcile', 'with code checks'])
arrow([(chk['r'], YA), (agg['l'], YA)])
vsv = node(162.5, YA, 20, 14, 'verify_severe', ['2nd model on severe', 'claims (v4.2-dev)'], 'llm', dashed=True)
arrow([(agg['r'], YA), (vsv['l'], YA)])

# ---------------------------------------------------------------- row B (y = 46): triage, persist, END
tri = node(120, YB, 20, 14, 'triage', ['escalate | pass with', 'findings | pass'])
per = node(150, YB, 18, 14, 'persist', ['agent record +', 'human decision'])
arrow([(tri['r'], YB), (per['l'], YB)], dashed=True)
label((tri['r'] + per['l']) / 2, YB + 2.4, 'otherwise', italic=True, size=4.8)
ax.add_patch(plt.Circle((172.0, YB), 2.1, fc='white', ec=INK, lw=LW * 1.2, zorder=3))
ax.add_patch(plt.Circle((172.0, YB), 1.2, color=INK, zorder=4))
ax.text(172.0, YB - 4.6, 'END', fontsize=5.0, color=GREY, ha='center', family=MONO)
arrow([(per['r'], YB), (172.0 - 2.1, YB)])

arrow([(vsv['x'], vsv['b']), (vsv['x'], 66.0), (tri['x'] + 5, 66.0), (tri['x'] + 5, tri['t'])])      # verify -> triage
arrow([(ext['x'], ext['b']), (ext['x'], 62.0), (tri['x'] - 5, 62.0), (tri['x'] - 5, tri['t'])], dashed=True)
label(ext['x'] + 1.2, 71.5, 'call failed', ha='left', italic=True)

cko = node(dx, YB, 21, 11.5, 'check_only', ['LLM only (ablation arm)'], 'llm')
arrow([(dx, YA - 4.6), (dx, cko['t'])], dashed=True)
label(dx + 1.2, (YA - 4.6 + cko['t']) / 2, 'model_only', ha='left', italic=True)
arrow([(cko['r'], YB), (tri['l'], YB)])

# ---------------------------------------------------------------- human in the loop and durable state (row H)
hum = node(120, YH, 25, 14, 'human_review', ['interrupt() for a clinician', 'accept | override | defer'], 'human')
arrow([(tri['x'], tri['b']), (tri['x'], hum['t'])], dashed=True)
label(tri['x'] + 1.2, (tri['b'] + hum['t']) / 2, 'escalate', ha='left', italic=True)
arrow([(hum['r'], YH), (per['x'], YH), (per['x'], per['b'])])
label((hum['r'] + per['x']) / 2, YH + 2.4, 'resume', italic=True)

cx, cy = 88.0, YH                                     # checkpoint store (cylinder)
ax.add_patch(Rectangle((cx - 6, cy - 4.2), 12, 8.4, fc='white', ec='none', zorder=3))
ax.plot([cx - 6, cx - 6], [cy - 4.2, cy + 4.2], color=CODE_E, lw=LW, zorder=4)
ax.plot([cx + 6, cx + 6], [cy - 4.2, cy + 4.2], color=CODE_E, lw=LW, zorder=4)
ax.add_patch(Ellipse((cx, cy + 4.2), 12, 2.6, fc='white', ec=CODE_E, lw=LW, zorder=4))
ax.add_patch(Ellipse((cx, cy - 4.2), 12, 2.6, fc='white', ec=CODE_E, lw=LW, zorder=2))
ax.text(cx, cy - 0.6, 'checkpoints', fontsize=5.0, color=INK, ha='center', va='center', zorder=5)
ax.plot([cx + 6, hum['l']], [cy, cy], color=LIGHT, lw=LW, ls=(0, (0.8, 1.0)), zorder=1)
ax.text((cx + 6 + hum['l']) / 2, cy + 1.6, 'pause', fontsize=4.8, color=GREY, ha='center', style='italic')
for i, t in enumerate(('SQLite, one thread per job', 'survives restarts', 'cancel | revise | recover')):
    ax.text(cx - 8.0, cy + 3.0 - i * 3.0, t, fontsize=5.0, color=GREY if i < 2 else INK, ha='right', va='center',
            family=MONO if i == 2 else SANS)

# ---------------------------------------------------------------- shared state strip
ax.add_patch(FancyBboxPatch((4, 1.6), W - 5, 8.6, boxstyle='round,pad=0,rounding_size=1.2', lw=LW * 0.8,
                            ec=LIGHT, fc='#F4F4F4', zorder=1))
ax.text(6.2, 5.9, 'Graph state', fontsize=5.6, weight='bold', color=INK, va='center')
for i, row in enumerate(('calls (reducer: append), cancellation (reducer: first), input_screen, extraction, problems,',
                         'unknown, rule_out, items, code_findings, check_result, findings, uncertain, triage_result, human_decision, record')):
    ax.text(23.0, 7.4 - i * 3.6, row, family=MONO, fontsize=4.6, color=GREY, va='center')
ax.text(W - 3.2, 3.8, 'each node: partial update + one trace span (no case text)', fontsize=4.8, color=GREY,
        va='center', ha='right', style='italic')

# ---------------------------------------------------------------- legend (top right)
LX, LY = 112.0, 118.0


def swatch(x, y, kind, text, dashed=False):
    edge, face = {'llm': (LLM_E, LLM_F), 'human': (HUM_E, HUM_F), 'code': (CODE_E, CODE_F)}[kind]
    ax.add_patch(FancyBboxPatch((x, y - 1.6), 5, 3.2, boxstyle='round,pad=0,rounding_size=0.8',
                                lw=LW * (1.4 if kind != 'code' else 1), ec=edge, fc=face,
                                ls=(0, (1.6, 1.0)) if dashed else '-'))
    ax.text(x + 6.5, y, text, fontsize=5.4, color=INK, va='center')


swatch(LX, LY, 'llm', 'LLM call (DeepSeek)')
swatch(LX + 36, LY, 'code', 'Deterministic code')
swatch(LX, LY - 5.5, 'human', 'Human in the loop')
swatch(LX + 36, LY - 5.5, 'code', 'v4.2-dev, off by default', dashed=True)
arrow([(LX, LY - 11), (LX + 5, LY - 11)])
ax.text(LX + 6.5, LY - 11, 'Static edge', fontsize=5.4, color=INK, va='center')
arrow([(LX + 36, LY - 11), (LX + 41, LY - 11)], dashed=True)
ax.text(LX + 42.5, LY - 11, 'Conditional edge', fontsize=5.4, color=INK, va='center')
ax.add_patch(Rectangle((LX + 1.9, LY - 18.5), 1.2, 4, color=INK))
ax.text(LX + 6.5, LY - 16.5, 'Fork / join', fontsize=5.4, color=INK, va='center')

OUT.mkdir(parents=True, exist_ok=True)
for ext_ in ('pdf', 'svg'):
    fig.savefig(OUT / f'v4-langgraph.{ext_}')
fig.savefig(OUT / 'v4-langgraph.png', dpi=600)
print('written', *(OUT / f'v4-langgraph.{e}' for e in ('pdf', 'svg', 'png')))
