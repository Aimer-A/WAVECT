# WaveCT repository rules

- The official workflow is inversion -> result rendering -> `wave_ct.validation_pipeline` -> report.
- Keep quantitative `velocity` separate from presentation-only `display_velocity`.
- A missing validation input must be reported as `SKIPPED`; never promote it to `PASS`.
- Checkerboard recovery measures numerical resolution only. It does not prove a real anomaly is geological.
- Travel-time alignment is not full-waveform validation. Full waveform claims require a source wavelet, material parameters, boundary conditions and solver configuration.
- Run Windows unit tests one test file per Python process to avoid duplicate OpenMP runtime conflicts.
- Treat Yanbei monthly cohorts as one independence group, not independent mines.
- Keep experimental DNR Fourier/differential settings at zero by default; promote a candidate only after fresh event-split validation, including a per-split degradation safety gate.
- Use `wave_ct.tools.multi_cohort_dnr_benchmark` for cross-source DNR promotion experiments.
