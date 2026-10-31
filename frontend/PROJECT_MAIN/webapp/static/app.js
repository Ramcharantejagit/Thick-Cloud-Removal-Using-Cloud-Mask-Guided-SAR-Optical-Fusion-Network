// CloudClear front-end: confirmations, run-form spinner, sample filter, charts (data from JSON blocks; no inline JS -> strict CSP)
(function () {
  const $ = (id) => document.getElementById(id);
  const data = (id) => { const el = $(id); return el ? JSON.parse(el.textContent) : null; };

  document.querySelectorAll('form[data-confirm]').forEach((f) =>
    f.addEventListener('submit', (e) => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));

  document.querySelectorAll('form.run-form').forEach((f) => f.addEventListener('submit', () => {
    const b = f.querySelector('button'); b.disabled = true;
    b.querySelector('.spinner-border')?.classList.remove('d-none');
  }));

  const filter = $('filter'), sel = $('patch');
  if (filter && sel) filter.addEventListener('input', () => {
    const q = filter.value.toLowerCase();
    [...sel.options].forEach((o) => { o.hidden = q && !o.text.toLowerCase().includes(q); });
  });

  if (typeof Chart === 'undefined') return;
  Chart.defaults.datasets.bar.maxBarThickness = 48;
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  const C = { blue: '#1d6f8c', orange: '#e07a1f', gray: '#9aa5b1', navy: '#0b2545', green: '#2a9d8f' };

  const series = data('userSeries');
  if (series && $('psnrChart')) {
    const s = series.filter((r) => r.psnr_out !== null);
    new Chart($('psnrChart'), { type: 'line', data: { labels: s.map((r, i) => '#' + (i + 1)),
      datasets: [{ label: 'Cloudy input', data: s.map((r) => r.psnr_in), borderColor: C.gray, backgroundColor: C.gray },
                 { label: 'Cloud-free prediction', data: s.map((r) => r.psnr_out), borderColor: C.blue, backgroundColor: C.blue }] },
      options: { plugins: { tooltip: { callbacks: { title: (it) => s[it[0].dataIndex].name } } }, scales: { y: { title: { display: true, text: 'PSNR (dB)' } } } } });
  }
  if (series && series.length && $('cloudChart')) {
    const bins = [0, 0, 0, 0, 0]; series.forEach((r) => bins[Math.min(4, Math.floor(r.cloud * 5))]++);
    new Chart($('cloudChart'), { type: 'bar', data: { labels: ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'],
      datasets: [{ label: 'Runs', data: bins, backgroundColor: C.blue }] }, options: { plugins: { legend: { display: false } }, scales: { y: { ticks: { precision: 0 } } } } });
  }
  const bench = data('benchData');
  if (bench && $('benchChart')) {
    new Chart($('benchChart'), { type: 'bar', data: { labels: bench.map((r) => r.Model.replace(' (no model)', '')),
      datasets: [{ label: 'Test PSNR (dB)', data: bench.map((r) => r.PSNR), backgroundColor: bench.map((r) => r.Model.startsWith('Cloudy') ? C.gray : C.blue) }] },
      options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { min: 15 } } } });
  }
  const reg = data('regionData');
  if (reg && $('regionChart')) {
    new Chart($('regionChart'), { type: 'bar', data: { labels: reg.map((r) => r.Model.replace(' (no model)', '')),
      datasets: [{ label: 'Cloud region', data: reg.map((r) => r.PSNR_cloud), backgroundColor: C.orange },
                 { label: 'Clear region', data: reg.map((r) => r.PSNR_clear), backgroundColor: C.blue }] },
      options: { scales: { y: { min: 15, title: { display: true, text: 'PSNR (dB)' } } } } });
  }
  const adm = data('adminData');
  if (adm && $('dayChart')) new Chart($('dayChart'), { type: 'bar', data: { labels: adm.per_day.map((r) => r.d),
    datasets: [{ label: 'Runs', data: adm.per_day.map((r) => r.n), backgroundColor: C.blue }] }, options: { plugins: { legend: { display: false } }, scales: { y: { ticks: { precision: 0 } } } } });
  if (adm && $('userChart')) new Chart($('userChart'), { type: 'bar', data: { labels: adm.per_user.map((r) => r.name),
    datasets: [{ label: 'Runs', data: adm.per_user.map((r) => r.n), backgroundColor: C.green }] }, options: { indexAxis: 'y', plugins: { legend: { display: false } }, scales: { x: { ticks: { precision: 0 } } } } });
})();
