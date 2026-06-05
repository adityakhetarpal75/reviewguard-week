import { generateTailoredResume, getSampleData } from "./tailor.js";

const form = document.querySelector("[data-tailor-form]");
const resumeInput = document.querySelector("#resume");
const jobInput = document.querySelector("#job");
const roleInput = document.querySelector("#target-role");
const companyInput = document.querySelector("#company-name");
const scoreValue = document.querySelector("[data-score-value]");
const scoreRing = document.querySelector("[data-score-ring]");
const coveredList = document.querySelector("[data-covered-list]");
const missingList = document.querySelector("[data-missing-list]");
const output = document.querySelector("#tailored-output");
const copyButton = document.querySelector("[data-copy-output]");
const downloadButton = document.querySelector("[data-download-output]");
const sampleButton = document.querySelector("[data-load-sample]");
const clearButton = document.querySelector("[data-clear]");
const statusMessage = document.querySelector("[data-status]");

let latestResult = null;

function setStatus(message, tone = "neutral") {
  statusMessage.textContent = message;
  statusMessage.dataset.tone = tone;
}

function setScore(score) {
  scoreValue.textContent = `${score}%`;
  scoreRing.style.setProperty("--score", `${score * 3.6}deg`);
}

function renderKeywords(container, keywords, emptyMessage) {
  container.replaceChildren();

  if (!keywords.length) {
    const empty = document.createElement("li");
    empty.className = "keyword keyword--empty";
    empty.textContent = emptyMessage;
    container.append(empty);
    return;
  }

  for (const keyword of keywords) {
    const item = document.createElement("li");
    item.className = "keyword";
    item.textContent = keyword.label;
    container.append(item);
  }
}

function updateButtons(hasOutput) {
  copyButton.disabled = !hasOutput;
  downloadButton.disabled = !hasOutput;
}

function runTailor() {
  const resume = resumeInput.value.trim();
  const job = jobInput.value.trim();

  if (!resume || !job) {
    latestResult = null;
    setScore(0);
    renderKeywords(coveredList, [], "Paste a resume and job description first");
    renderKeywords(missingList, [], "Paste a resume and job description first");
    output.value = "";
    updateButtons(false);
    setStatus("Add both documents to generate a tailored resume package.");
    return;
  }

  latestResult = generateTailoredResume(resume, job, {
    targetRole: roleInput.value.trim(),
    companyName: companyInput.value.trim()
  });

  setScore(latestResult.analysis.score);
  renderKeywords(coveredList, latestResult.analysis.covered.slice(0, 10), "No covered keywords yet");
  renderKeywords(missingList, latestResult.analysis.missing.slice(0, 10), "No major gaps detected");
  output.value = latestResult.text;
  updateButtons(true);

  const tone = latestResult.analysis.score >= 70 ? "success" : latestResult.analysis.score >= 40 ? "warning" : "neutral";
  setStatus("Tailored package generated. Review suggested additions for accuracy before sending.", tone);
}

async function copyOutput() {
  if (!latestResult) {
    return;
  }

  try {
    await navigator.clipboard.writeText(latestResult.text);
    setStatus("Copied tailored output to clipboard.", "success");
  } catch {
    output.select();
    document.execCommand("copy");
    setStatus("Copied tailored output to clipboard.", "success");
  }
}

function downloadOutput() {
  if (!latestResult) {
    return;
  }

  const file = new Blob([latestResult.text], { type: "text/plain" });
  const link = document.createElement("a");
  const roleSlug = roleInput.value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  link.href = URL.createObjectURL(file);
  link.download = `${roleSlug || "tailored-resume"}.txt`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
  setStatus("Downloaded tailored resume package.", "success");
}

function loadSample() {
  const sample = getSampleData();
  resumeInput.value = sample.resume;
  jobInput.value = sample.job;
  roleInput.value = "Senior Product Manager";
  companyInput.value = "Acme Software";
  runTailor();
}

function clearForm() {
  form.reset();
  runTailor();
  resumeInput.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runTailor();
});

copyButton.addEventListener("click", copyOutput);
downloadButton.addEventListener("click", downloadOutput);
sampleButton.addEventListener("click", loadSample);
clearButton.addEventListener("click", clearForm);

for (const input of [resumeInput, jobInput, roleInput, companyInput]) {
  input.addEventListener("input", () => {
    if (latestResult) {
      runTailor();
    }
  });
}

runTailor();
