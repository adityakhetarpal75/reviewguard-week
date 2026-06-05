# Resume Tailor

A browser-based resume tailoring app that helps candidates adapt a resume to a job description without sending data to a server.

## Features

- Paste a base resume and job description.
- See a match score, covered keywords, and missing keywords.
- Generate a tailored summary, prioritized resume bullets, skills suggestions, and a cover letter draft.
- Copy or download the tailored output as plain text.
- Runs entirely in the browser with deterministic keyword analysis.

## Run locally

```bash
cd resume-tailor
npm start
```

Then open <http://localhost:4173>.

You can also open `index.html` directly in a browser, though serving the directory is recommended for ES module loading.

## Test

```bash
cd resume-tailor
npm test
```

## Privacy note

All tailoring is done in the browser. The app does not make network requests or store resume content outside the current page session.
