# does-ai-regulation-work
Does AI regulation actually reduce harm? A causal timing analysis of policy events, incidents, and model releases across 138 countries (2022-2025).

A causal timing analysis asking the question AI governance debates usually skip past: does regulation actually change anything, or is a lot of it just activity without results?

Every year brings a new wave of AI laws, ethics boards, and regulatory frameworks — and a parallel wave of headlines about deepfakes, AI fraud, and algorithmic harm. The two are treated as connected, but rarely tested against each other directly. If policy isn't measurably reducing harm, that's not a footnote — it's the difference between governance that protects people and governance that just looks like it does.

## Why this project

This project uses five linked global datasets (2022-2025) to test policy against outcomes directly, rather than assuming a law "worked" because it passed. It's built around four questions, worked in this order:

1. **Does harm tend to happen before policy responds, or does policy ever get ahead of it?**
2. **Do specific policy events — deepfake laws specifically — produce a measurable before/after change in related incidents?**
3. **Does a wave of image/video-generation model releases predict a rise in deepfake-related incidents afterward?**
4. **Which countries show "regulation theater"** — a lot of policy activity with no accompanying improvement in public trust?

## Data

[Global GenAI Adoption and Impact Index (2022-2025)](https://www.kaggle.com/datasets/aamir28/global-genai-adoption-and-impact-index-2022-2025) — five linked tables:

| Table | Grain | Rows | Covers |
|---|---|---|---|
| `gaaii_country_year.csv` | country × year | 552 | Adoption, regulation stringency, harm indices, public trust |
| `gaaii_incidents.csv` | individual incident | 1,800 | Deepfakes, AI hiring bias, fraud, misinformation, with severity and sector |
| `gaaii_policy_events.csv` | individual policy event | 520 | Laws, bills, treaties, with lobbying/opposition scores |
| `gaaii_model_releases.csv` | individual model | 70 | 25 organizations, release dates, categories |
| `gaaii_survey_microdata.csv` | individual respondent | 22,989 | Public trust, fear of job loss, experienced harm (not used directly in these four questions — available for follow-up work) |

None of these tables share a single common key at the same grain — `incidents` and `policy_events` are event-level, `country_year` is annual, `model_releases` is global. Most of the analytical work in this project is building the connective tissue: rolling incidents up to country-quarter counts, converting dates into a comparable sortable timeline, and merging tables that don't naturally line up.

## Key findings

| Question | Finding |
|---|---|
| **Does harm precede policy?** | No clear pattern. Incidents were nearly flat in the 2 quarters before vs. after a policy event (6.12 vs. 6.52 average), and slightly more events showed incidents *rising* afterward (295) than falling (194). |
| **Do deepfake laws reduce deepfake incidents?** | Narrowing to a clean, category-matched comparison didn't change the story — 1.0 incidents before vs. 1.3 after on average, and only 9 of 30 countries (30%) saw a decrease after the law passed. |
| **Do model releases predict incident spikes?** | The one place a real signal showed up: a moderate positive correlation (r = 0.44) between visual-generation model releases in a quarter and deepfake-related incidents the following quarter — flagged honestly as suggestive, not proof, given the small sample (12-16 quarters of data). |
| **"Regulation theater" countries** | Seven countries show high policy activity with declining public trust: India, UAE, Turkey, Italy, France, Poland, and China. India and UAE lead in policy event count (27 each); UAE shows the steepest trust decline (-6.81 points). The group spans multiple income levels, so this isn't purely a wealth story. |

The throughline across all four questions is stronger than any single one: **policy activity in this dataset doesn't reliably translate into either less harm or more public trust**, at least not on the timelines and measures tested here.

## Implications — why this matters beyond the notebook

If these patterns held up in real-world policy evaluation (with more rigorous causal methods and better data than this dataset can offer), the practical takeaway for anyone working in AI governance, trust and safety, or public policy would be blunt: **passing a law is not the same as solving the problem it names.** A few directions this points toward:

- **Measurement, not just legislation, needs to be the deliverable.** Every policy event in this dataset could plausibly generate a natural experiment if governments tracked and published outcome metrics before and after enactment — right now, that evaluation step is largely missing, which is exactly why a project like this has to reconstruct it after the fact instead of pointing to an official report.
- **The model-release-to-incident link (Question 3) suggests harm prevention may need to happen upstream, at the model-release stage, not downstream at the legal-response stage** — by the time a law passes in response to a harm category, the tooling that enables it is often already widely available.
- **"Regulation theater" is a real, identifiable pattern, not just a cynical talking point.** Being able to name it with a specific, repeatable method (policy activity vs. trust outcome, not just activity vs. vibes) is the kind of evidence a Trust & Safety or governance team could actually act on — flagging which jurisdictions need outcome-focused reform, not just more bills introduced.

## Repo structure

```
genai-trust-safety-policy-analysis/
├── genai_trust_safety_analysis.ipynb    # full notebook
├── genai_trust_safety_analysis.py       # matching script version
├── gaaii_country_year.csv
├── gaaii_incidents.csv
├── gaaii_policy_events.csv
├── gaaii_model_releases.csv
├── gaaii_survey_microdata.csv
└── README.md
```

The `.py` and `.ipynb` versions are kept in sync — the notebook is generated directly from the script.

## How to run

1. Clone the repo and make sure all five CSVs are in the same folder as the notebook.
2. Install the required packages:
   ```
   pip install pandas
   ```
3. Open `genai_trust_safety_analysis.ipynb` in Jupyter and run all cells top to bottom.

## Tools

- **pandas** — the entire analysis: multi-table merging, groupby rollups, custom time-window logic for the before/after comparisons, and the quarter-arithmetic groundwork that makes the timing questions possible in the first place

## What's next

- Bring in `gaaii_survey_microdata.csv` — does individual-level experience of AI harm predict personal trust more strongly than country-level policy activity does?
- Extend Question 3's model-release analysis beyond correlation — a proper lagged regression or Granger-causality style test would say more than a single correlation coefficient can.
- Widen Question 2 beyond deepfake laws to test other clean policy-to-incident category matches (e.g. "AI Bias in Hiring" incidents against hiring-specific regulation, echoing the causal hiring bias project).
- Revisit the ±2 quarter window used throughout — a sensitivity check with wider or narrower windows would show whether these findings are robust to that choice or an artifact of it.

## Author

Keshia Neal, Ph.D.
