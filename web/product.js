'use strict';

// A viewer for two saved CLI comparisons. No model calls or browser-side gate verdicts.
const byId = (id) => document.getElementById(id);
const sourceTitles = {
  'leave-current': 'Parental leave policy', 'keys-current': 'Production credentials policy',
  'retention-current': 'Approved retention policy', 'retention-superseded': '2022 proposal',
  expenses: 'Expense claims policy', 'refund-current': 'Approved refund policy',
  'refund-superseded': 'Ended refund pilot', 'refund-hard-negative': 'Refund notifications',
  'leave-hard-negative': 'Annual leave requests', 'incident-handbook': 'Incident handbook',
  'oncall-rota': 'On-call rota',
};
const shortExcerpts = {
  'retention-current': 'Delete raw customer event records after 30 days.',
  'retention-superseded': '365 days · never approved.',
};
let evidence;
let scenario = 'regression';
let selected = 2;
const text = (tag, content, className) => {
  const element = document.createElement(tag);
  element.textContent = content;
  if (className) element.className = className;
  return element;
};
function sourceState(id, entry) {
  if (id.endsWith('-superseded')) return ['Superseded', 'superseded'];
  if (entry.relevant_ids.includes(id)) return [id.endsWith('-current') ? 'Current' : 'Relevant source', 'current'];
  if (entry.required_context_ids.includes(id)) return ['Required context', 'context'];
  return ['Other source', 'other'];
}
function renderRanking(target, entry) {
  const rows = entry.retrieved_ids.slice(0, 2).map((id, index) => {
    const row = document.createElement('li');
    const rank = text('span', String(index + 1), 'rank');
    rank.setAttribute('aria-hidden', 'true');
    const body = document.createElement('div');
    const heading = document.createElement('div');
    heading.className = 'source-heading';
    heading.append(text('h4', sourceTitles[id] || id));
    const [label, tone] = sourceState(id, entry);
    heading.append(text('span', label, `source-state ${tone}`));
    const source = evidence.documents.find((doc) => doc.id === id);
    body.append(heading, text('p', shortExcerpts[id] || (source.text.length > 170 ? `${source.text.slice(0, 170)}…` : source.text)));
    row.append(rank, body);
    return row;
  });
  byId(target).replaceChildren(...rows);
}
function renderSources(before, after) {
  const container = byId('source-evidence');
  const sections = [[before, 'Before update'], [after, 'After update']].map(([entry, label]) => {
    const section = document.createElement('section');
    section.append(text('h3', `${label} · all ${entry.retrieved_ids.length} retrieved sources`));
    const list = document.createElement('ol');
    for (const id of entry.retrieved_ids) {
      const item = document.createElement('li');
      const source = evidence.documents.find((doc) => doc.id === id);
      item.append(text('h4', sourceTitles[id] || id), text('p', source.text), text('code', id));
      list.append(item);
    }
    section.append(list);
    return section;
  });
  container.replaceChildren(...sections);
}
function render() {
  const report = evidence.scenarios[scenario];
  const before = report.baseline_report.cases[selected];
  const after = report.candidate_report.cases[selected];
  byId('question').textContent = before.query;
  byId('position').textContent = `${selected + 1} of ${report.baseline_report.cases.length}`;
  byId('question-select').value = String(selected);
  byId('previous').disabled = selected === 0;
  byId('next').disabled = selected === report.baseline_report.cases.length - 1;
  renderRanking('before-sources', before);
  renderRanking('after-sources', after);
  renderSources(before, after);
  const retentionSwap = !report.passed && before.id === 'conflicting-retention-evidence';
  document.querySelector('.rank-swap').hidden = !retentionSwap;
  byId('verdict').textContent = report.passed ? 'No regression detected' : 'Hold this update';
  byId('decision').classList.toggle('no-regression', report.passed);
  if (retentionSwap) {
    byId('consequence').textContent = 'The current policy is still found. The outdated proposal now ranks above it.';
  } else if (before.id === 'refund-window-with-conflict') {
    byId('consequence').textContent = 'An ended refund pilot ranks first in both runs. No new drop here does not mean the existing result is good.';
  } else if (before.id === 'sev1-escalation' && !report.passed) {
    byId('consequence').textContent = 'The incident handbook stays first. Other sources change order; the ranking score for this question does not fall.';
  } else {
    byId('consequence').textContent = report.passed
      ? 'This question keeps the same source order. Passing this small test does not prove the assistant is ready to release.'
      : 'This question keeps the same source order. The update is held because data-retention evidence ranks worse.';
  }
  const count = report.candidate_report.cases.filter((entry) => entry.passed).length;
  byId('case-summary').textContent = `All ${count} case checks pass. ${report.passed ? 'No ranking drop was detected in this test.' : 'Evidence ranking still worsens.'}`;
  byId('switch-result').textContent = report.passed ? 'Switch to regressing update' : 'Switch to unchanged ranking';
  const stem = report.passed ? 'unchanged' : 'comparison';
  byId('download').href = `${stem}.json`;
  byId('full-report').href = report.passed ? 'unchanged-report.html' : 'report.html';
  byId('reproduce').textContent = report.reproduce;
  byId('run-name').textContent = report.candidate_report.config.name;
  const metricRows = report.metrics.filter((metric) => ['mrr_at_k', 'ndcg_at_k'].includes(metric.metric));
  byId('metric-summary').replaceChildren(...metricRows.map((metric) => text('p', `${metric.metric === 'mrr_at_k' ? 'Rank of the first relevant source (MRR)' : 'Quality of the source order (nDCG)'}: ${metric.baseline.toFixed(4)} before, ${metric.candidate.toFixed(4)} after. ${metric.verdict}.`)));
  const changed = before.retrieved_ids.join('|') !== after.retrieved_ids.join('|');
  byId('question-note').textContent = `Showing the first two sources. ${changed ? 'The full ranking changed.' : 'The full ranking is unchanged.'} Open the evidence below for every retrieved source and its exact text.`;
  byId('announcement').textContent = `${selected + 1} of 6. ${before.query} ${byId('verdict').textContent}. ${byId('consequence').textContent}`;
}
function changeQuestion(index) {
  selected = index;
  render();
}
async function start() {
  try {
    const response = await fetch('evidence.json');
    if (!response.ok) throw new Error('Saved evidence is unavailable');
    const data = await response.json();
    // Fail closed on a damaged deployment. The CLI validates the committed evidence in full.
    if (data.version !== 1 || !Array.isArray(data.documents)) throw new Error('Invalid evidence bundle');
    for (const key of ['regression', 'unchanged']) {
      const report = data.scenarios?.[key];
      if (!report || typeof report.passed !== 'boolean' || !Array.isArray(report.metrics)
        || report.baseline_report?.cases?.length !== 6 || report.candidate_report?.cases?.length !== 6) throw new Error('Incomplete saved comparison');
      for (const entry of [...report.baseline_report.cases, ...report.candidate_report.cases]) {
        if (!Array.isArray(entry.retrieved_ids) || entry.retrieved_ids.some((id) => !data.documents.some((doc) => doc.id === id && typeof doc.text === 'string')))
          throw new Error('Missing source evidence');
      }
    }
    evidence = data;
    render();
    byId('previous').addEventListener('click', () => changeQuestion(selected - 1));
    byId('next').addEventListener('click', () => changeQuestion(selected + 1));
    byId('question-select').addEventListener('change', (event) => changeQuestion(Number(event.target.value)));
    byId('switch-result').addEventListener('click', () => {
      scenario = scenario === 'regression' ? 'unchanged' : 'regression';
      render();
      if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
        for (const row of byId('after-sources').children) row.animate([{ transform: 'translateY(8px)' }, { transform: 'translateY(0)' }], { duration: 280, easing: 'cubic-bezier(.16,1,.3,1)' });
      }
    });
    byId('switch-result').hidden = false;
    byId('switch-result').disabled = false;
    byId('question-picker').hidden = false;
    byId('next').disabled = false;
    byId('previous').disabled = false;
    document.body.classList.add('interactive');
  } catch {
    byId('load-error').hidden = false;
    document.querySelector('.question-navigation').hidden = true;
  }
}
start();
