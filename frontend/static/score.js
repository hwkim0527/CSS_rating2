const form = document.getElementById("score-form");
const resultBox = document.getElementById("result");
const resetBtn = document.getElementById("reset-btn");
const submitBtn = document.getElementById("submit-btn");

// model 은 점수 산출 입력이 아니라 엔드포인트 선택용이므로 payload 에서 제외한다.
const numericFields = new Set([
  "loan_amnt", "installment", "annual_inc",
  "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
  "revol_bal", "revol_util", "total_acc", "mort_acc",
  "pub_rec_bankruptcies", "credit_history_years",
]);

const ENDPOINTS = { xgboost: "/api/score", llm: "/api/score_llm" };

// 페이지 로드 시 LLM 사용 가능 여부를 확인해 라디오를 켜고/끈다.
// GPU 없는 호스트(Render/HF CPU/Cloud Run 기본)에서는 LLM 이 비활성이므로
// 사용자가 고르지 못하게 막고 이유를 안내한다(graceful degradation).
async function refreshLlmStatus() {
  const note = document.getElementById("llm-note");
  const card = document.getElementById("llm-card");
  const radio = card.querySelector('input[name="model"]');
  try {
    const res = await fetch("/api/llm_status");
    if (!res.ok) throw new Error(res.statusText);
    const s = await res.json();
    if (s.available_for_inference) {
      note.textContent = "AI 언어모델 · 사용 가능";
      radio.disabled = false;
      card.classList.remove("disabled");
    } else {
      radio.disabled = true;
      card.classList.add("disabled");
      note.textContent = s.enabled_flag
        ? "어댑터 준비 중 · 현재 비활성"
        : "GPU 환경에서만 활성 (현재 XGBoost만 사용 가능)";
    }
  } catch (_e) {
    radio.disabled = true;
    card.classList.add("disabled");
    note.textContent = "상태 확인 실패 · XGBoost만 사용 가능";
  }
}
refreshLlmStatus();

function selectedModel() {
  const checked = form.querySelector('input[name="model"]:checked');
  return checked ? checked.value : "xgboost";
}

function setLoading(on, model) {
  if (on) {
    submitBtn.disabled = true;
    submitBtn.textContent = model === "llm"
      ? "AI 분석 중… (모델 로딩 시 수십 초 소요)"
      : "평가 중…";
  } else {
    submitBtn.disabled = false;
    submitBtn.textContent = "신용점수 산출하기";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const model = selectedModel();
  const endpoint = ENDPOINTS[model] || ENDPOINTS.xgboost;

  const data = {};
  const fd = new FormData(form);
  for (const [k, v] of fd.entries()) {
    if (k === "model") continue; // 엔드포인트 선택용 — 점수 입력 아님
    if (numericFields.has(k)) data[k] = Number(v);
    else data[k] = v;
  }

  setLoading(true, model);
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      if (res.status === 503 && model === "llm") {
        alert(
          "LLM(sLLM) 평가를 사용할 수 없습니다.\n" +
          "이 모델은 GPU 환경에서만 동작합니다. XGBoost 로 다시 시도하세요.\n\n" +
          `상세: ${err.detail || res.statusText}`
        );
      } else {
        alert(`평가 실패: ${err.detail || res.statusText}`);
      }
      return;
    }
    renderResult(await res.json());
  } catch (err) {
    alert(`요청 실패: ${err.message}`);
  } finally {
    setLoading(false, model);
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  resultBox.classList.add("hidden");
  refreshLlmStatus(); // reset 이 라디오를 기본값으로 되돌리므로 안내문도 갱신
});

function renderResult(out) {
  const prob = Number(out.default_probability) || 0;
  document.getElementById("r-score").textContent = out.credit_score;
  document.getElementById("r-grade").textContent = out.risk_grade;   // 등급 문자(badge 가 "등급" 표기)
  document.getElementById("r-prob").innerHTML =
    (prob * 100).toFixed(1) + '<span class="pct">%</span>';
  document.getElementById("r-model").textContent = out.model_name;
  document.getElementById("r-grade-kr").textContent = out.risk_grade_kr;

  // 라디얼 게이지 채움 + 등급별 위험색(A~E) 부여
  const gauge = document.getElementById("r-gauge");
  gauge.style.setProperty("--p", Math.max(0.02, Math.min(prob, 1)).toFixed(3));
  resultBox.className = "result card grade-" + (out.risk_grade || "C");

  const factors = document.getElementById("r-factors");
  factors.innerHTML = "";
  for (const f of out.top_factors) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="tag impact-${f.impact}">${f.impact}</span>` +
      `<span><strong>${f.factor}</strong> · ${f.note}</span>`;
    factors.appendChild(li);
  }
  resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
}
