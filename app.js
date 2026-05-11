/**
 * app.js
 * Lógica principal de la aplicación Horse Colic Predictor
 * Depende de: models/lr_model.js, models/nn_model.js
 */

// ─── Estado global ───────────────────────────────────────────────────────────
let currentModel      = 'lr';   // modelo activo en predicción individual
let currentBatchModel = 'lr';   // modelo activo en predicción por lotes
let uploadedData      = null;   // filas del CSV cargado

// ─── Switching de tabs ────────────────────────────────────────────────────────
function switchTab(tabId, clickedBtn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  clickedBtn.classList.add('active');
}

// ─── Selección de modelo ──────────────────────────────────────────────────────
function setModel(m, btn) {
  currentModel = m;
  document.querySelectorAll('#tab-individual .model-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function setBatchModel(m, btn) {
  currentBatchModel = m;
  document.querySelectorAll('#tab-batch .model-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// ─── Predicción individual ────────────────────────────────────────────────────
function getFormFeatures() {
  return Array.from({ length: 22 }, (_, i) => {
    const el = document.getElementById('f' + (i + 1));
    const v  = parseFloat(el.value);
    return isNaN(v) ? NaN : v;
  });
}

function predict(features, model) {
  return model === 'lr' ? lrPredict(features) : nnPredict(features);
}

function predictIndividual() {
  const features = getFormFeatures();
  const result   = predict(features, currentModel);
  const isSurg   = result.surgery >= 0.5;

  const badge = document.getElementById('result-badge');
  badge.textContent  = isSurg ? '🔴 Requiere Cirugía' : '🟢 No Requiere Cirugía';
  badge.className    = 'result-badge ' + (isSurg ? 'surgery' : 'no-surgery');

  document.getElementById('result-desc').textContent = isSurg
    ? 'El modelo indica que este paciente probablemente necesita intervención quirúrgica. Se recomienda evaluación clínica inmediata.'
    : 'El modelo indica que este paciente probablemente puede tratarse sin cirugía. Continúe con monitoreo clínico.';

  document.getElementById('result-model-tag').textContent =
    currentModel === 'lr' ? 'Regresión Logística' : 'Red Neuronal';

  document.getElementById('prob-surgery-val').textContent   = (result.surgery   * 100).toFixed(1) + '%';
  document.getElementById('prob-no-surgery-val').textContent = (result.noSurgery * 100).toFixed(1) + '%';

  // Animación del card
  const card = document.getElementById('result-card');
  card.classList.remove('show');
  void card.offsetHeight; // forzar reflow
  card.classList.add('show');

  // Animar barras con pequeño delay
  setTimeout(() => {
    document.getElementById('prob-surgery-bar').style.width   = (result.surgery   * 100) + '%';
    document.getElementById('prob-no-surgery-bar').style.width = (result.noSurgery * 100) + '%';
  }, 60);
}

// ─── Carga de archivo (Batch) ─────────────────────────────────────────────────
function handleFileSelect(e) { loadFile(e.target.files[0]); }

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-area').classList.remove('drag-over');
  loadFile(e.dataTransfer.files[0]);
}
function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('upload-area').classList.add('drag-over');
}
function handleDragLeave() {
  document.getElementById('upload-area').classList.remove('drag-over');
}

function loadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    let lines = e.target.result.trim().split('\n').filter(l => l.trim());

    // Saltar encabezado si la primera fila no es numérica
    if (lines.length && isNaN(parseFloat(lines[0].trim().split(/[\s,]+/)[0]))) {
      lines = lines.slice(1);
    }

    uploadedData = lines.map(l =>
      l.trim().split(/[\s,]+/).map(v => (v === '?' ? NaN : parseFloat(v)))
    );
    document.getElementById('file-info').classList.add('show');
    document.getElementById('file-name-display').textContent = file.name;
    document.getElementById('file-meta-display').textContent = uploadedData.length + ' registros cargados';
    document.getElementById('batch-predict-btn').style.display = 'inline-block';
    document.getElementById('batch-results').classList.remove('show');
  };
  reader.readAsText(file);
}

function removeFile() {
  uploadedData = null;
  document.getElementById('file-info').classList.remove('show');
  document.getElementById('batch-predict-btn').style.display = 'none';
  document.getElementById('batch-results').classList.remove('show');
  document.getElementById('file-input').value = '';
}

// ─── Predicción por lotes ─────────────────────────────────────────────────────
function predictBatch() {
  if (!uploadedData || uploadedData.length === 0) return;

  const results = uploadedData.map(row => {
    const features  = row.slice(0, 22);
    const rawLabel  = row[22]; // columna 23 del CSV (índice 22): 1=cirugía, 2=no cirugía
    const pred      = predict(features, currentBatchModel);
    const predClass = pred.surgery >= 0.5 ? 1 : 0;
    const trueClass = isNaN(rawLabel) ? null : (rawLabel === 1 ? 1 : 0);
    return { pred, predClass, trueClass, features };
  });

  const hasLabels = results.some(r => r.trueClass !== null);

  _renderMetrics(results, hasLabels);
  _renderConfusionMatrix(results, hasLabels);
  _renderClassReport(results, hasLabels);
  _renderPreviewTable(results, hasLabels);

  const container = document.getElementById('batch-results');
  container.classList.remove('show');
  void container.offsetHeight;
  container.classList.add('show');
}

// ── helpers internos ──
function _calcMetrics(results) {
  const labeled = results.filter(r => r.trueClass !== null);
  let tp = 0, tn = 0, fp = 0, fn = 0;
  labeled.forEach(r => {
    if      (r.predClass === 1 && r.trueClass === 1) tp++;
    else if (r.predClass === 0 && r.trueClass === 0) tn++;
    else if (r.predClass === 1 && r.trueClass === 0) fp++;
    else                                              fn++;
  });
  const total = labeled.length;
  const acc   = total ? (tp + tn) / total : 0;
  const prec  = (tp + fp) ? tp / (tp + fp) : 0;
  const rec   = (tp + fn) ? tp / (tp + fn) : 0;
  const f1    = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0;
  return { tp, tn, fp, fn, acc, prec, rec, f1, total };
}

function _renderMetrics(results, hasLabels) {
  const el = document.getElementById('metrics-row');
  if (!hasLabels) {
    el.innerHTML = `<div class="info-box" style="grid-column:1/-1">
      Sin etiquetas reales detectadas. Para ver métricas incluye la columna 24 (attr_24) en el archivo.
    </div>`;
    return;
  }
  const { acc, prec, rec, f1 } = _calcMetrics(results);
  el.innerHTML = [
    ['Accuracy',   (acc  * 100).toFixed(1) + '%'],
    ['Precisión',  (prec * 100).toFixed(1) + '%'],
    ['Recall',     (rec  * 100).toFixed(1) + '%'],
    ['F1-Score',   f1.toFixed(3)]
  ].map(([label, val]) => `
    <div class="metric-card">
      <div class="value">${val}</div>
      <div class="label">${label}</div>
    </div>`).join('');
}

function _renderConfusionMatrix(results, hasLabels) {
  const el = document.getElementById('cm-grid');
  if (!hasLabels) { el.innerHTML = '<p style="color:var(--muted);font-size:.82rem">Sin etiquetas disponibles</p>'; return; }
  const { tp, tn, fp, fn } = _calcMetrics(results);
  el.innerHTML = `
    <div></div>
    <div class="cm-header">Pred: Cirugía</div>
    <div class="cm-header">Pred: No Cirugía</div>
    <div class="cm-label">Real: Cirugía</div>
    <div class="cm-cell cm-tp">${tp}</div>
    <div class="cm-cell cm-fn">${fn}</div>
    <div class="cm-label">Real: No Cirugía</div>
    <div class="cm-cell cm-fp">${fp}</div>
    <div class="cm-cell cm-tn">${tn}</div>`;
}

function _renderClassReport(results, hasLabels) {
  const tbody = document.getElementById('report-body');
  if (!hasLabels) { tbody.innerHTML = ''; return; }
  const { tp, tn, fp, fn, total } = _calcMetrics(results);

  const prec1  = (tp + fp) ? tp / (tp + fp) : 0;
  const rec1   = (tp + fn) ? tp / (tp + fn) : 0;
  const f1_1   = (prec1 + rec1) ? 2 * prec1 * rec1 / (prec1 + rec1) : 0;

  const prec0  = (tn + fn) ? tn / (tn + fn) : 0;
  const rec0   = (tn + fp) ? tn / (tn + fp) : 0;
  const f1_0   = (prec0 + rec0) ? 2 * prec0 * rec0 / (prec0 + rec0) : 0;

  const avgP   = (prec0 + prec1) / 2;
  const avgR   = (rec0  + rec1)  / 2;
  const avgF   = (f1_0  + f1_1)  / 2;

  tbody.innerHTML = `
    <tr>
      <td class="class-name">🟢 No Cirugía</td>
      <td>${(prec0*100).toFixed(1)}%</td>
      <td>${(rec0*100).toFixed(1)}%</td>
      <td>${f1_0.toFixed(3)}</td>
      <td>${tn + fp}</td>
    </tr>
    <tr>
      <td class="class-name">🔴 Cirugía</td>
      <td>${(prec1*100).toFixed(1)}%</td>
      <td>${(rec1*100).toFixed(1)}%</td>
      <td>${f1_1.toFixed(3)}</td>
      <td>${tp + fn}</td>
    </tr>
    <tr>
      <td class="class-name muted">Promedio macro</td>
      <td>${(avgP*100).toFixed(1)}%</td>
      <td>${(avgR*100).toFixed(1)}%</td>
      <td>${avgF.toFixed(3)}</td>
      <td>${total}</td>
    </tr>`;
}

function _renderPreviewTable(results, hasLabels) {
  document.getElementById('real-col-header').style.display  = hasLabels ? '' : 'none';
  document.getElementById('match-col-header').style.display = hasLabels ? '' : 'none';

  const tbody = document.getElementById('preview-body');
  tbody.innerHTML = results.slice(0, 100).map((r, i) => {
    const fmt  = v => (isNaN(v) ? '—' : v);
    const isSurg = r.predClass === 1;
    let extra = '';
    if (hasLabels && r.trueClass !== null) {
      const realLbl = r.trueClass === 1
        ? '<span class="pred-surgery">Cirugía</span>'
        : '<span class="pred-no-surgery">No Cirugía</span>';
      extra = `<td>${realLbl}</td><td>${r.predClass === r.trueClass ? '✅' : '❌'}</td>`;
    }
    return `<tr>
      <td class="muted">${i + 1}</td>
      <td>${fmt(r.features[3])}</td>
      <td>${fmt(r.features[4])}</td>
      <td>${fmt(r.features[10])}</td>
      <td class="${isSurg ? 'pred-surgery' : 'pred-no-surgery'}">${isSurg ? '🔴 Cirugía' : '🟢 No Cirugía'}</td>
      <td class="mono">${(r.pred.surgery * 100).toFixed(1)}%</td>
      ${extra}
    </tr>`;
  }).join('');
}
