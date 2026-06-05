import assert from "node:assert/strict";
import test from "node:test";
import { analyzeResume, extractKeywords, generateTailoredResume, getSampleData, splitResumeSections } from "../src/tailor.js";

test("extractKeywords prioritizes repeated and specific job terms", () => {
  const keywords = extractKeywords(
    "Own product strategy, roadmap prioritization, SQL analytics, dashboard analytics, and customer research.",
    8
  );
  const labels = keywords.map((keyword) => keyword.phrase);

  assert.ok(labels.includes("analytics"));
  assert.ok(labels.some((label) => label.includes("product strategy")));
  assert.ok(labels.some((label) => label.includes("customer research")));
});

test("analyzeResume reports covered and missing keyword groups", () => {
  const resume = "Skills\nSQL, analytics, customer research\nExperience\n- Built analytics dashboards with SQL.";
  const job = "Need SQL analytics, roadmap prioritization, customer research, and experimentation.";
  const analysis = analyzeResume(resume, job, 8);

  assert.ok(analysis.score > 0);
  assert.ok(analysis.covered.some((keyword) => keyword.phrase === "analytics"));
  assert.ok(analysis.missing.some((keyword) => keyword.phrase.includes("roadmap")));
});

test("splitResumeSections detects common resume sections", () => {
  const sections = splitResumeSections("Jane Doe\n\nProfessional Summary\nBuilder\n\nSkills\nPython, SQL");

  assert.deepEqual(
    sections.map((section) => section.name),
    ["header", "summary", "skills"]
  );
});

test("generateTailoredResume returns actionable package text", () => {
  const sample = getSampleData();
  const result = generateTailoredResume(sample.resume, sample.job, {
    targetRole: "Senior Product Manager",
    companyName: "Acme Software"
  });

  assert.match(result.text, /TAILORED RESUME PACKAGE/);
  assert.match(result.text, /Professional Summary/);
  assert.match(result.text, /Cover Letter Draft/);
  assert.ok(result.analysis.score >= 40);
});
