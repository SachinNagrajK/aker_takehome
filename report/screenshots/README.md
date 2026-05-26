# Screenshots for REPORT.tex

Drop PNG files into this folder using the exact filenames below. Order matches the
figure order in `REPORT.tex` so swapping any one image is a single file replace.

| #  | Filename                          | What to capture                                                                                              | Where to take it                                                                            |
|----|-----------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| 1  | `01-chat-kpi-chart.png`           | A finished answer that includes a KPI card *and* a chart (e.g. "Show me the rent trend for 115r over 12 months"). | Main chat UI                                                                                |
| 2  | `02-tool-trace.png`               | The expanded tool-trace panel showing 2–3 tool calls with their per-tool durations.                          | Click the trace icon on a recent assistant message                                          |
| 3  | `03-clarification.png`            | The clarification card asking for property or time scope.                                                    | Ask an ambiguous question like "what's the occupancy?" with no property selected            |
| 4  | `04-monitoring-runs.png`          | Monitoring tab listing recent eval runs with mean scores.                                                    | `Monitoring` tab after running `/evals/runs` at least once                                  |
| 5  | `05-monitoring-run-detail.png`    | A single run drilled in — per-case score, tool calls, link to Phoenix trace.                                 | Click into one of the rows on the Monitoring tab                                            |
| 6  | `06-phoenix-trace.png`            | A Phoenix Cloud trace tree for one chat turn (`extract_scope` → `agent` → `tools` → `compose`).              | The Phoenix project page after one `/chat` call with `PHOENIX_ENABLED=true`                 |

The system-design diagram (figure 8 in the report) is hand-drawn in TikZ inside
`REPORT.tex` — no screenshot needed.

## Recommended capture settings

- Use 2× / Retina resolution if possible; LaTeX will scale to width.
- Crop tightly — the report includes them at full text width and side-by-side at
  half width (Figures 2/3 and 6/7).
- PNG, not JPG — the UI is mostly flat colour and will compress better.
