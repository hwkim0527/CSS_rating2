// Pure-canvas bar chart for AUC (no external chart library).
(function () {
  const canvas = document.getElementById("auc-chart");
  if (!canvas || !window.METRICS || !window.METRICS.models) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  const models = window.METRICS.models;
  const items = [];
  for (const key of Object.keys(models)) {
    const m = models[key];
    items.push({ name: m.label_kr || key, value: m.auc });
  }

  // Background (paper)
  ctx.fillStyle = "#fbf8f1";
  ctx.fillRect(0, 0, W, H);

  const paddingLeft = 200;
  const paddingTop = 20;
  const paddingBottom = 30;
  const paddingRight = 30;
  const chartW = W - paddingLeft - paddingRight;
  const chartH = H - paddingTop - paddingBottom;
  const barH = Math.min(30, chartH / items.length - 8);
  const gap = (chartH - barH * items.length) / (items.length + 1);

  // Axes (vertical at 0.5 = random)
  const xFor = (v) => paddingLeft + (v / 1.0) * chartW;

  // Reference grid
  ctx.strokeStyle = "#ddd5c4";
  ctx.fillStyle = "#8a8170";
  ctx.font = "11px 'JetBrains Mono', monospace";
  for (let t = 0.5; t <= 1.0; t += 0.1) {
    const x = xFor(t);
    ctx.beginPath();
    ctx.moveTo(x, paddingTop);
    ctx.lineTo(x, H - paddingBottom);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), x - 8, H - 12);
  }

  // Bars
  items.forEach((it, i) => {
    const y = paddingTop + gap + i * (barH + gap);
    const value = it.value;
    // Label
    ctx.fillStyle = "#1b1813";
    ctx.font = "12px 'Pretendard', sans-serif";
    const label = it.name.length > 26 ? it.name.slice(0, 25) + "…" : it.name;
    ctx.fillText(label, 10, y + barH / 2 + 4);

    if (value == null) {
      ctx.fillStyle = "#ddd5c4";
      ctx.fillRect(paddingLeft, y, 10, barH);
      ctx.fillStyle = "#8a8170";
      ctx.fillText("학습 대기", paddingLeft + 16, y + barH / 2 + 4);
      return;
    }

    const startX = xFor(0.5);
    const endX = xFor(value);
    const grad = ctx.createLinearGradient(startX, 0, endX, 0);
    grad.addColorStop(0, "#2f9b76");
    grad.addColorStop(1, "#154a39");
    ctx.fillStyle = grad;
    ctx.fillRect(startX, y, endX - startX, barH);

    ctx.fillStyle = "#1b1813";
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    ctx.fillText(value.toFixed(4), endX + 6, y + barH / 2 + 4);
  });

  // Title
  ctx.fillStyle = "#1b1813";
  ctx.font = "bold 13px 'Noto Serif KR', serif";
  ctx.fillText("Test-set AUC (0.5=무작위, 1.0=완벽)", 10, 14);
})();
