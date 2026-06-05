const STOP_WORDS = new Set([
  "a",
  "about",
  "above",
  "across",
  "after",
  "again",
  "against",
  "all",
  "also",
  "am",
  "an",
  "and",
  "any",
  "are",
  "as",
  "at",
  "be",
  "because",
  "been",
  "before",
  "being",
  "below",
  "between",
  "both",
  "but",
  "by",
  "can",
  "candidate",
  "company",
  "could",
  "did",
  "do",
  "does",
  "doing",
  "during",
  "each",
  "few",
  "for",
  "from",
  "further",
  "had",
  "has",
  "have",
  "having",
  "he",
  "her",
  "here",
  "hers",
  "herself",
  "him",
  "himself",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "itself",
  "job",
  "just",
  "looking",
  "me",
  "more",
  "most",
  "need",
  "my",
  "myself",
  "no",
  "nor",
  "not",
  "now",
  "of",
  "off",
  "on",
  "once",
  "only",
  "or",
  "other",
  "our",
  "ours",
  "responsibilities",
  "ourselves",
  "out",
  "over",
  "own",
  "per",
  "include",
  "includes",
  "ideal",
  "role",
  "same",
  "she",
  "should",
  "so",
  "some",
  "such",
  "than",
  "that",
  "the",
  "their",
  "theirs",
  "them",
  "themselves",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "through",
  "to",
  "too",
  "under",
  "until",
  "up",
  "very",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "while",
  "who",
  "whom",
  "why",
  "will",
  "with",
  "within",
  "would",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves"
]);

const SECTION_ALIASES = {
  summary: ["summary", "professional summary", "profile", "objective"],
  experience: ["experience", "work experience", "professional experience", "employment", "work history"],
  skills: ["skills", "technical skills", "core competencies", "competencies"],
  projects: ["projects", "selected projects", "portfolio"],
  education: ["education", "academic background"],
  certifications: ["certifications", "certificates", "licenses"]
};

const TECH_HINTS = new Set([
  "agile",
  "analytics",
  "api",
  "aws",
  "azure",
  "budgeting",
  "ci",
  "cloud",
  "crm",
  "css",
  "dashboard",
  "data",
  "devops",
  "docker",
  "excel",
  "figma",
  "forecasting",
  "git",
  "html",
  "javascript",
  "jira",
  "kpi",
  "kubernetes",
  "linux",
  "marketing",
  "metrics",
  "node",
  "python",
  "react",
  "reporting",
  "roadmap",
  "salesforce",
  "scrum",
  "seo",
  "sql",
  "strategy",
  "typescript",
  "ux"
]);

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\w+#/-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizePhrase(phrase) {
  return normalizeText(phrase)
    .split(" ")
    .filter(Boolean)
    .join(" ");
}

function tokenize(text) {
  return normalizeText(text)
    .split(" ")
    .filter((token) => token.length > 1 && !STOP_WORDS.has(token) && !/^\d+$/.test(token));
}

function titleCase(phrase) {
  return phrase
    .split(" ")
    .map((word) => {
      if (word.length <= 3 && word === word.toLowerCase()) {
        return word.toUpperCase();
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

function hasPhrase(text, phrase) {
  if (!phrase) {
    return false;
  }
  const normalizedText = ` ${normalizeText(text)} `;
  const normalizedPhrase = normalizePhrase(phrase);
  if (!normalizedPhrase) {
    return false;
  }
  if (normalizedText.includes(` ${normalizedPhrase} `)) {
    return true;
  }

  const phraseWords = normalizedPhrase.split(" ");
  if (phraseWords.length === 1) {
    return false;
  }

  const textWords = new Set(tokenize(text));
  return phraseWords.every((word) => textWords.has(word));
}

function phraseSpecificity(phrase) {
  const words = phrase.split(" ");
  const techBonus = words.some((word) => TECH_HINTS.has(word)) ? 1.2 : 0;
  const lengthBonus = Math.min(words.length * 0.9, 2.7);
  const uncommonBonus = words.some((word) => word.length >= 8) ? 0.6 : 0;
  return lengthBonus + techBonus + uncommonBonus;
}

export function extractKeywords(text, limit = 20) {
  const counts = new Map();
  const segments = String(text || "")
    .split(/[\n\r,.;:()]+/)
    .map((segment) => tokenize(segment))
    .filter((segmentTokens) => segmentTokens.length);

  for (const tokens of segments) {
    for (let size = 1; size <= 2; size += 1) {
      for (let index = 0; index <= tokens.length - size; index += 1) {
        const phraseTokens = tokens.slice(index, index + size);
        if (phraseTokens.some((token) => token.length < 3 && !["ai", "ui", "ux"].includes(token))) {
          continue;
        }

        const phrase = phraseTokens.join(" ");
        counts.set(phrase, (counts.get(phrase) || 0) + 1);
      }
    }
  }

  const candidates = [...counts.entries()]
    .filter(([phrase, count]) => {
      const words = phrase.split(" ");
      return words.length > 1 || count > 1 || TECH_HINTS.has(phrase) || phrase.length >= 7;
    })
    .map(([phrase, count]) => ({
      phrase,
      label: titleCase(phrase),
      count,
      score: count * 2 + phraseSpecificity(phrase)
    }))
    .sort(
      (a, b) =>
        b.score - a.score ||
        b.phrase.split(" ").length - a.phrase.split(" ").length ||
        b.phrase.length - a.phrase.length
    );

  const selected = [];
  for (const candidate of candidates) {
    const isDuplicate = selected.some((keyword) => {
      const existingWords = keyword.phrase.split(" ");
      const candidateWords = candidate.phrase.split(" ");
      return (
        keyword.phrase.includes(candidate.phrase) ||
        candidate.phrase.includes(keyword.phrase) ||
        candidateWords.every((word) => existingWords.includes(word))
      );
    });

    if (!isDuplicate) {
      selected.push(candidate);
    }

    if (selected.length >= limit) {
      break;
    }
  }

  return selected;
}

export function splitResumeSections(resumeText) {
  const sections = [];
  const lines = String(resumeText || "").split(/\r?\n/);
  let current = { name: "header", title: "Header", lines: [] };

  const findSectionName = (line) => {
    const cleaned = normalizePhrase(line.replace(/[:|-]+$/g, ""));
    if (!cleaned || cleaned.length > 40) {
      return null;
    }

    for (const [name, aliases] of Object.entries(SECTION_ALIASES)) {
      if (aliases.includes(cleaned)) {
        return name;
      }
    }
    return null;
  };

  for (const line of lines) {
    const sectionName = findSectionName(line);
    if (sectionName) {
      if (current.lines.some((entry) => entry.trim())) {
        sections.push(current);
      }
      current = { name: sectionName, title: line.trim().replace(/[:|-]+$/g, ""), lines: [] };
    } else {
      current.lines.push(line);
    }
  }

  if (current.lines.some((entry) => entry.trim())) {
    sections.push(current);
  }

  return sections;
}

export function extractBullets(resumeText) {
  const bulletPattern = /^\s*(?:[-*•]|\d+[.)])\s+/;
  return String(resumeText || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => bulletPattern.test(line))
    .map((line) => line.replace(bulletPattern, "").trim())
    .filter(Boolean);
}

export function analyzeResume(resumeText, jobText, keywordLimit = 18) {
  const keywords = extractKeywords(jobText, keywordLimit);
  const covered = [];
  const missing = [];

  for (const keyword of keywords) {
    if (hasPhrase(resumeText, keyword.phrase)) {
      covered.push(keyword);
    } else {
      missing.push(keyword);
    }
  }

  const score = keywords.length ? Math.round((covered.length / keywords.length) * 100) : 0;
  const bullets = extractBullets(resumeText);
  const prioritizedBullets = bullets
    .map((bullet) => ({
      text: bullet,
      matches: keywords.filter((keyword) => hasPhrase(bullet, keyword.phrase)),
      score: keywords.reduce((total, keyword) => total + (hasPhrase(bullet, keyword.phrase) ? keyword.score : 0), 0)
    }))
    .sort((a, b) => b.score - a.score || b.text.length - a.text.length);

  return {
    score,
    keywords,
    covered,
    missing,
    sections: splitResumeSections(resumeText),
    bullets,
    prioritizedBullets
  };
}

function getFirstUsefulLine(section) {
  if (!section) {
    return "";
  }
  return section.lines.map((line) => line.trim()).find((line) => line && !/^[-*•]/.test(line)) || "";
}

function buildSummary(resumeText, analysis, targetRole) {
  const summarySection = analysis.sections.find((section) => section.name === "summary");
  const currentSummary = getFirstUsefulLine(summarySection);
  const role = targetRole || "the target role";
  const coveredLabels = analysis.covered.slice(0, 4).map((keyword) => keyword.label);
  const missingLabels = analysis.missing.slice(0, 3).map((keyword) => keyword.label);

  if (currentSummary) {
    const emphasis = coveredLabels.length
      ? ` Emphasize proven experience with ${coveredLabels.join(", ")}.`
      : " Emphasize the most relevant outcomes from your existing experience.";
    const additions = missingLabels.length
      ? ` If accurate, weave in ${missingLabels.join(", ")} because they appear in the job description.`
      : "";
    return `${currentSummary}${emphasis}${additions}`;
  }

  const resumeSignals = extractKeywords(resumeText, 5).map((keyword) => keyword.label);
  const signalText = resumeSignals.length ? ` with background in ${resumeSignals.join(", ")}` : "";
  const keywordText = coveredLabels.length ? ` Aligns existing experience with ${coveredLabels.join(", ")}.` : "";
  const additionText = missingLabels.length
    ? ` Add evidence for ${missingLabels.join(", ")} only where it reflects real experience.`
    : "";
  return `Professional targeting ${role}${signalText}.${keywordText}${additionText}`.replace(/\s+/g, " ").trim();
}

function formatKeywordList(keywords, fallback = "None identified") {
  return keywords.length ? keywords.map((keyword) => keyword.label).join(", ") : fallback;
}

function buildCoverLetter(options, analysis) {
  const role = options.targetRole || "the open role";
  const company = options.companyName || "your team";
  const strengths = formatKeywordList(analysis.covered.slice(0, 4), "the qualifications in the job description");
  const focus = formatKeywordList(analysis.missing.slice(0, 3), "the role's highest-priority requirements");

  return [
    `Dear Hiring Team,`,
    ``,
    `I am excited to apply for ${role} at ${company}. My resume shows relevant experience across ${strengths}, and I am especially interested in contributing to the priorities described in your posting.`,
    ``,
    `In this version of my application, I have highlighted the work that best maps to the role and called attention to measurable outcomes from my prior experience. I would welcome the chance to discuss how that background can support ${company}'s goals.`,
    ``,
    `Before sending, I will verify that any added emphasis around ${focus} is supported by accurate examples from my work.`,
    ``,
    `Sincerely,`,
    `[Your Name]`
  ].join("\n");
}

export function generateTailoredResume(resumeText, jobText, options = {}) {
  const analysis = analyzeResume(resumeText, jobText);
  const roleLine = options.targetRole ? `Target role: ${options.targetRole}` : "Target role: Not specified";
  const companyLine = options.companyName ? `Company: ${options.companyName}` : "Company: Not specified";
  const topBullets = analysis.prioritizedBullets.filter((bullet) => bullet.score > 0).slice(0, 8);
  const fallbackBullets = analysis.prioritizedBullets.slice(0, 5);
  const bulletsToUse = topBullets.length ? topBullets : fallbackBullets;

  const tailoredBullets = bulletsToUse.length
    ? bulletsToUse.map((bullet) => {
        const matched = bullet.matches.length ? ` (${bullet.matches.map((keyword) => keyword.label).join(", ")})` : "";
        return `- ${bullet.text}${matched}`;
      })
    : ["- Add 3-5 achievement bullets from your resume that directly support this job description."];

  const sectionNames = analysis.sections
    .filter((section) => !["header", "summary"].includes(section.name))
    .map((section) => section.title)
    .filter(Boolean);

  return {
    analysis,
    text: [
      "TAILORED RESUME PACKAGE",
      roleLine,
      companyLine,
      "",
      "Match Score",
      `${analysis.score}% keyword coverage (${analysis.covered.length}/${analysis.keywords.length} priority terms found in the resume).`,
      "",
      "Professional Summary",
      buildSummary(resumeText, analysis, options.targetRole),
      "",
      "Priority Keywords Already Covered",
      formatKeywordList(analysis.covered),
      "",
      "Keywords To Add If Accurate",
      formatKeywordList(analysis.missing, "No major gaps detected"),
      "",
      "Experience Bullets To Move Up Or Emphasize",
      ...tailoredBullets,
      "",
      "Recommended Resume Structure",
      "- Header",
      "- Professional Summary tailored to this role",
      "- Skills section with covered keywords first",
      "- Experience bullets ordered by relevance to the job",
      ...(sectionNames.length ? sectionNames.map((name) => `- Keep ${name}`) : ["- Education and certifications as applicable"]),
      "",
      "Cover Letter Draft",
      buildCoverLetter(options, analysis),
      "",
      "Final Review Checklist",
      "- Keep only keywords you can defend with real examples.",
      "- Add metrics to the top 3 bullets where possible.",
      "- Mirror the employer's wording for accurate skills and responsibilities.",
      "- Remove lower-relevance bullets if the resume runs longer than the target length."
    ].join("\n")
  };
}

export function getSampleData() {
  return {
    resume: [
      "Jordan Lee",
      "Product Manager",
      "jordan@example.com | 555-0100 | Austin, TX",
      "",
      "Professional Summary",
      "Product manager with experience launching customer-facing software and coordinating cross-functional teams.",
      "",
      "Skills",
      "Roadmapping, user research, stakeholder management, analytics, SQL, agile delivery",
      "",
      "Experience",
      "- Led a cross-functional team of engineering, design, and support partners to launch a self-service onboarding flow.",
      "- Used SQL dashboards and customer interviews to identify activation bottlenecks and prioritize roadmap work.",
      "- Improved onboarding completion by 18% by simplifying setup steps and rewriting in-product guidance.",
      "- Partnered with sales and customer success to package feedback into quarterly planning themes.",
      "",
      "Education",
      "B.A. Business Administration"
    ].join("\n"),
    job: [
      "Senior Product Manager",
      "",
      "We are looking for a senior product manager to own product strategy, roadmap prioritization, analytics, experimentation, and cross-functional execution.",
      "The ideal candidate has experience with customer research, SQL, dashboard metrics, go-to-market planning, stakeholder communication, and agile software delivery.",
      "Responsibilities include defining product requirements, measuring KPIs, improving onboarding conversion, and partnering with engineering, design, sales, and customer success."
    ].join("\n")
  };
}
