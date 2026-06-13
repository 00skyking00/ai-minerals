"""Builds the interactive Tiger-style animation widget for Chapter 7.

Reads the trajectory JSON from
`data/derived/bcgt/v20_marker_animation_data.json`, inlines it as a
JS const, and writes a Quarto partial at
`portfolio/_widget_v20_marker_animation.qmd` containing the HTML/CSS/JS
in a `{=html}` block. The chapter includes the partial via
`{{< include _widget_v20_marker_animation.qmd >}}`.

Run after `v20_marker_animation_data.py` (or whenever the trajectory
data changes).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_JSON = REPO / "data/derived/bcgt/v20_marker_animation_data.json"
OUT_QMD = REPO / "portfolio/_widget_v20_marker_animation.qmd"


WIDGET_TEMPLATE = r"""```{=html}
<style>
.bcgt-anim-root {
  font-family: var(--bs-body-font-family, system-ui, sans-serif);
  max-width: 100%;
  margin: 1.5em 0;
  padding: 1em;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #fafafa;
}
.bcgt-anim-root h5.bcgt-title {
  margin: 0 0 .25em 0;
  font-size: 1em;
  font-weight: 600;
}
.bcgt-anim-root h6.bcgt-policy-label {
  margin: 0 0 .35em 0;
  font-size: .95em;
  font-weight: 600;
  color: #111827;
}
.bcgt-panels {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1em;
  margin: 1em 0;
}
@media (max-width: 640px) {
  .bcgt-panels { grid-template-columns: 1fr; }
}
.bcgt-panel {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: .75em;
}
.bcgt-belief {
  display: flex;
  height: 18px;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: .5em;
  font-size: .72em;
  font-weight: 600;
  color: white;
  min-width: 0;
}
.bcgt-belief-A, .bcgt-belief-B {
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  white-space: nowrap;
  transition: flex .35s ease;
}
.bcgt-belief-A { background: #ec4899; }
.bcgt-belief-B { background: #8b5cf6; }
.bcgt-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  margin-bottom: .5em;
}
.bcgt-cell {
  aspect-ratio: 1;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: .78em;
  font-weight: 500;
  color: #374151;
  position: relative;
  border: 3px solid transparent;
  transition: border-color .2s, transform .15s;
  text-align: center;
  padding: 4px;
}
.bcgt-cell.kind-claim { background: #fef3c7; }
.bcgt-cell.kind-marker { background: #dbeafe; }
.bcgt-cell.current {
  border-color: #dc2626;
  transform: scale(1.04);
}
.bcgt-cell-badge {
  position: absolute;
  top: 2px;
  right: 4px;
  background: rgba(31,41,55,0.85);
  color: white;
  border-radius: 50%;
  min-width: 18px;
  height: 18px;
  font-size: .7em;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0 4px;
}
.bcgt-cell-obs {
  position: absolute;
  bottom: 2px;
  right: 4px;
  font-size: .9em;
  font-weight: 700;
}
.bcgt-cell-obs.obs-1 { color: #16a34a; }
.bcgt-cell-obs.obs-0 { color: #6b7280; }
.bcgt-reward {
  font-size: 1em;
  font-weight: 600;
  margin-top: .25em;
}
.bcgt-reward.pos { color: #16a34a; }
.bcgt-reward.neg { color: #dc2626; }
.bcgt-caption {
  font-size: .8em;
  color: #4b5563;
  margin-top: .35em;
  min-height: 3em;
}
.bcgt-controls {
  display: flex;
  gap: .5em;
  align-items: center;
  flex-wrap: wrap;
}
.bcgt-controls select,
.bcgt-controls button {
  padding: .35em .7em;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  background: white;
  font-size: .85em;
  cursor: pointer;
  font-family: inherit;
}
.bcgt-controls button:hover { background: #f3f4f6; }
.bcgt-controls button.primary {
  background: #1f2937;
  color: white;
  border-color: #1f2937;
}
.bcgt-controls button.primary:hover { background: #374151; }
.bcgt-step-counter {
  font-size: .85em;
  color: #4b5563;
}
.bcgt-truth-reveal {
  margin-top: .5em;
  padding: .5em .75em;
  background: #f3f4f6;
  border-radius: 4px;
  font-size: .85em;
  color: #4b5563;
}
.bcgt-legend {
  display: flex;
  gap: 1em;
  font-size: .78em;
  color: #4b5563;
  margin-top: .35em;
  flex-wrap: wrap;
}
.bcgt-legend-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  margin-right: 4px;
  vertical-align: middle;
}
</style>

<div class="bcgt-anim-root" id="bcgt-anim-root">
  <h5 class="bcgt-title">Step-through: how each policy plays the Tiger problem</h5>
  <div class="bcgt-controls">
    <select id="bcgt-ep" aria-label="episode"></select>
    <button id="bcgt-back" aria-label="step backward">&larr; back</button>
    <button id="bcgt-play" class="primary" aria-label="play or pause">Play</button>
    <button id="bcgt-fwd" aria-label="step forward">forward &rarr;</button>
    <button id="bcgt-reset" aria-label="reset">reset</button>
    <span class="bcgt-step-counter" id="bcgt-step-counter">Step 0</span>
  </div>
  <div class="bcgt-legend">
    <span><span class="bcgt-legend-swatch" style="background:#fef3c7"></span>claim cell: pays +50 right, -30 wrong; no info</span>
    <span><span class="bcgt-legend-swatch" style="background:#dbeafe"></span>marker cell: no reward; informative reading</span>
    <span><span class="bcgt-legend-swatch" style="background:#ec4899"></span>belief P(Hypothesis A)</span>
    <span><span class="bcgt-legend-swatch" style="background:#8b5cf6"></span>belief P(Hypothesis B)</span>
  </div>
  <div class="bcgt-panels" id="bcgt-panels"></div>
  <div class="bcgt-truth-reveal" id="bcgt-truth-reveal"></div>
</div>

<script>
(function() {
  const BCGT_DATA = __BCGT_DATA__;
  const POLICIES = ['random', 'greedy_MAP', 'pomcp', 'sarsop'];
  const POLICY_LABELS = {
    random: 'Random (control)',
    greedy_MAP: 'Greedy MAP',
    pomcp: 'POMCP (online MCTS)',
    sarsop: 'SARSOP (offline alpha-vectors)',
  };

  let currentEpisode = 0;
  let currentStep = 0;
  let playInterval = null;
  const PLAY_INTERVAL_MS = 1400;

  const panelsEl = document.getElementById('bcgt-panels');
  const epSelect = document.getElementById('bcgt-ep');
  const playBtn = document.getElementById('bcgt-play');
  const backBtn = document.getElementById('bcgt-back');
  const fwdBtn = document.getElementById('bcgt-fwd');
  const resetBtn = document.getElementById('bcgt-reset');
  const stepCounter = document.getElementById('bcgt-step-counter');
  const truthReveal = document.getElementById('bcgt-truth-reveal');

  BCGT_DATA.episodes.forEach((ep, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Episode ${i+1}: ${ep.label}`;
    epSelect.appendChild(opt);
  });

  function actionLabel(cellIdx) {
    return BCGT_DATA.experiment.cell_labels[cellIdx];
  }

  function prettyTruth(t) {
    return t.replace('H_', 'Hypothesis ');
  }

  function createPanels() {
    panelsEl.innerHTML = '';
    POLICIES.forEach(p => {
      const panel = document.createElement('div');
      panel.className = 'bcgt-panel';
      panel.dataset.policy = p;
      const cells = BCGT_DATA.experiment.cell_layout.map((c, i) => `
        <div class="bcgt-cell kind-${c.kind}" data-idx="${i}">
          ${BCGT_DATA.experiment.cell_labels[i]}
          <span class="bcgt-cell-badge"></span>
          <span class="bcgt-cell-obs"></span>
        </div>
      `).join('');
      panel.innerHTML = `
        <h6 class="bcgt-policy-label">${POLICY_LABELS[p]}</h6>
        <div class="bcgt-belief">
          <div class="bcgt-belief-A">A 50%</div>
          <div class="bcgt-belief-B">B 50%</div>
        </div>
        <div class="bcgt-grid">${cells}</div>
        <div class="bcgt-reward">Reward: +0.00</div>
        <div class="bcgt-caption">Uniform prior. No drills yet.</div>
      `;
      panelsEl.appendChild(panel);
    });
  }

  function renderPanel(p, steps, ep) {
    const panel = panelsEl.querySelector(`[data-policy="${p}"]`);

    let belief, drilledMap, lastAction, totalReward, caption;
    if (currentStep === 0) {
      belief = steps[0].belief_before;
      drilledMap = {};
      lastAction = null;
      totalReward = 0;
      caption = 'Belief: 50/50. No drills yet.';
    } else {
      const s = steps[currentStep - 1];
      belief = s.belief_after;
      drilledMap = {};
      for (let t = 0; t < currentStep; t++) {
        const action = steps[t].action;
        if (!drilledMap[action]) {
          drilledMap[action] = {count: 0, lastObs: null};
        }
        drilledMap[action].count++;
        drilledMap[action].lastObs = steps[t].observation;
      }
      lastAction = s.action;
      totalReward = s.cumulative_reward;
      const aBefore = Math.round(s.belief_before[0] * 100);
      const aAfter = Math.round(s.belief_after[0] * 100);
      const obsTxt = s.observation === 1 ? 'positive' : 'negative';
      let actionTxt = `Drilled ${actionLabel(s.action)}`;
      if (s.already_drilled) actionTxt += ' (re-drill)';
      caption = `${actionTxt}, got a ${obsTxt} reading. ` +
                `P(Hypothesis A): ${aBefore}% &rarr; ${aAfter}%.`;
    }

    const beliefBar = panel.querySelector('.bcgt-belief');
    const aPct = Math.round(belief[0] * 100);
    const bPct = Math.round(belief[1] * 100);
    const aDiv = beliefBar.querySelector('.bcgt-belief-A');
    const bDiv = beliefBar.querySelector('.bcgt-belief-B');
    aDiv.style.flex = `${Math.max(0.0001, belief[0])}`;
    bDiv.style.flex = `${Math.max(0.0001, belief[1])}`;
    aDiv.textContent = `A ${aPct}%`;
    bDiv.textContent = `B ${bPct}%`;

    panel.querySelectorAll('.bcgt-cell').forEach(cell => {
      const idx = parseInt(cell.dataset.idx, 10);
      const isCurrent = (idx === lastAction);
      cell.classList.toggle('current', isCurrent);
      const badge = cell.querySelector('.bcgt-cell-badge');
      const obs = cell.querySelector('.bcgt-cell-obs');
      const info = drilledMap[idx];
      if (info) {
        badge.style.display = 'flex';
        badge.textContent = `${info.count}`;
        obs.textContent = info.lastObs === 1 ? '+' : '0';
        obs.className = `bcgt-cell-obs obs-${info.lastObs}`;
      } else {
        badge.style.display = 'none';
        obs.textContent = '';
        obs.className = 'bcgt-cell-obs';
      }
    });

    const rewardEl = panel.querySelector('.bcgt-reward');
    const sign = totalReward >= 0 ? '+' : '';
    rewardEl.textContent = `Reward: ${sign}${totalReward.toFixed(2)}`;
    rewardEl.className = `bcgt-reward ${totalReward > 0 ? 'pos' : totalReward < 0 ? 'neg' : ''}`;

    panel.querySelector('.bcgt-caption').innerHTML = caption;
  }

  function render() {
    const ep = BCGT_DATA.episodes[currentEpisode];
    stepCounter.textContent =
      `Drill ${currentStep} of ${BCGT_DATA.experiment.drill_budget}`;

    if (currentStep === 0) {
      truthReveal.innerHTML =
        `Each policy starts with a uniform 50/50 prior over Hypothesis A vs B. ` +
        `Ground truth this episode: <b>${prettyTruth(ep.truth_label)}</b>. ` +
        `The policies do not see this; they have to infer it.`;
    } else {
      truthReveal.innerHTML = `Ground truth: <b>${prettyTruth(ep.truth_label)}</b>.`;
    }

    POLICIES.forEach(p => renderPanel(p, ep.policies[p].steps, ep));
  }

  function stepForward() {
    if (currentStep < BCGT_DATA.experiment.drill_budget) {
      currentStep++;
      render();
      if (currentStep >= BCGT_DATA.experiment.drill_budget) {
        pause();
      }
    }
  }

  function stepBack() {
    if (currentStep > 0) {
      currentStep--;
      render();
    }
  }

  function play() {
    if (currentStep >= BCGT_DATA.experiment.drill_budget) {
      currentStep = 0;
      render();
    }
    if (playInterval) clearInterval(playInterval);
    playInterval = setInterval(stepForward, PLAY_INTERVAL_MS);
    playBtn.textContent = 'Pause';
  }

  function pause() {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
    playBtn.textContent = 'Play';
  }

  function reset() {
    pause();
    currentStep = 0;
    render();
  }

  function changeEpisode() {
    currentEpisode = parseInt(epSelect.value, 10);
    reset();
  }

  playBtn.addEventListener('click', () => {
    if (playInterval) pause(); else play();
  });
  fwdBtn.addEventListener('click', () => { pause(); stepForward(); });
  backBtn.addEventListener('click', () => { pause(); stepBack(); });
  resetBtn.addEventListener('click', reset);
  epSelect.addEventListener('change', changeEpisode);

  createPanels();
  render();
})();
</script>
```
"""


def main() -> int:
    with open(DATA_JSON) as f:
        data = json.load(f)
    inlined = json.dumps(data, separators=(",", ":"))
    out = WIDGET_TEMPLATE.replace("__BCGT_DATA__", inlined)
    OUT_QMD.write_text(out)
    print(f"wrote {OUT_QMD} ({len(out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
