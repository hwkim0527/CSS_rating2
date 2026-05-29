const form = document.getElementById("score-form");
const resultBox = document.getElementById("result");
const resetBtn = document.getElementById("reset-btn");

const numericFields = new Set([
  "loan_amnt", "installment", "int_rate", "annual_inc",
  "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
  "revol_bal", "revol_util", "total_acc", "mort_acc",
  "pub_rec_bankruptcies", "credit_history_years",
]);

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = {};
  const fd = new FormData(form);
  for (const [k, v] of fd.entries()) {
    if (numericFields.has(k)) data[k] = Number(v);
    else data[k] = v;
  }
  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`평가 실패: ${err.detail || res.statusText}`);
      return;
    }
    const out = await res.json();
    renderResult(out);
  } catch (err) {
    alert(`요청 실패: ${err.message}`);
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  resultBox.classList.add("hidden");
});

function renderResult(out) {
  document.getElementById("r-score").textContent = out.credit_score;
  document.getElementById("r-grade").textContent = `등급 ${out.risk_grade}`;
  document.getElementById("r-prob").textContent = (out.default_probability * 100).toFixed(2) + "%";
  document.getElementById("r-model").textContent = out.model_name;
  document.getElementById("r-grade-kr").textContent = out.risk_grade_kr;

  const factors = document.getElementById("r-factors");
  factors.innerHTML = "";
  for (const f of out.top_factors) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="impact-${f.impact}">[${f.impact}]</span> <strong>${f.factor}</strong>: ${f.note}`;
    factors.appendChild(li);
  }
  resultBox.classList.remove("hidden");
  resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
}
