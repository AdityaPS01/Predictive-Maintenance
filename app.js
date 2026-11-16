document.addEventListener('DOMContentLoaded', () => {
  // Stagger cards into view while respecting reduced-motion preferences.
  const cards = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in-view');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    cards.forEach(card => observer.observe(card));
  }

  const form = document.getElementById('prediction-form');
  if (!form) return;
  const panel = document.getElementById('result-panel');
  const title = document.getElementById('result-title');
  const label = panel.querySelector('.result-label');
  const description = document.getElementById('result-description');
  const probabilityValue = document.getElementById('probability-value');
  const probabilityBar = document.getElementById('probability-bar');

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button[type="submit"]');
    const oldText = button.innerHTML;
    button.disabled = true;
    button.textContent = 'Analyzing sensor profile…';
    panel.classList.remove('is-failure', 'is-safe');
    label.textContent = 'PROCESSING';
    title.textContent = 'Evaluating inputs';
    description.textContent = 'Applying the selected model to the sensor readings.';
    probabilityValue.textContent = '—';
    probabilityBar.style.width = '0%';
    try {
      const response = await fetch('/api/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.fromEntries(new FormData(form).entries()))
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Prediction could not be completed.');
      const failed = result.prediction === 1;
      panel.classList.add(failed ? 'is-failure' : 'is-safe');
      label.textContent = failed ? 'ELEVATED CLASSIFICATION' : 'CLASSIFICATION COMPLETE';
      title.textContent = result.message;
      description.textContent = `${result.model} classified the supplied sensor profile as ${failed ? 'failure (1)' : 'no failure (0)'}.`;
      if (result.estimated_probability === null || result.estimated_probability === undefined) {
        probabilityValue.textContent = 'Unavailable';
      } else {
        const pct = Math.max(0, Math.min(100, result.estimated_probability * 100));
        probabilityValue.textContent = `${pct.toFixed(1)}%`;
        probabilityBar.style.width = `${pct}%`;
      }
    } catch (error) {
      label.textContent = 'REQUEST ERROR';
      title.textContent = 'Unable to predict';
      description.textContent = error.message;
    } finally {
      button.disabled = false;
      button.innerHTML = oldText;
    }
  });
});
