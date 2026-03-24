# ORM Native Readiness Checklist

## Goal

Move ORM decision logic to native with this architecture:

- `BE -> Python adapter -> Native core`

Python remains the contract boundary. Native does not read BE JSON directly.

## Decisions Already Locked

- BE sends the full ORM config.
- Python adapter parses, validates, normalizes, and maps config to native input.
- Native core handles only core ORM logic.
- `selection_mode` comes from config and is attached to each question via ROI data.

## ORM Rule v1 Locked

- Single question with multiple filled bubbles -> `invalid_multiple_on_single`
- Multiple question with 2+ strong filled bubbles -> `multiple`
- Multiple question with exactly 1 filled bubble -> `single`
- Multiple question with 0 filled bubbles -> `blank`
- Borderline second bubble on single question -> `uncertain`
- Borderline recovered second bubble on multiple question -> `uncertain`
- `invalid_multiple_on_single` and `uncertain` both map to `answers = -1`

## Native Scope for v1

Native should return only core fields:

- `answers`
- `selected_options`
- `question_statuses`

Python adapter can derive:

- `multiple_questions`
- `invalid_multiple_on_single_questions`
- `blank_questions`
- `single_questions`
- `status_counts`

## Readiness Checklist

- [ ] Confirm the BE payload shape for ROI layout and `selection_mode`
- [ ] Freeze the Python adapter normalization rules
- [x] Define the native input contract for question bubbles and thresholds
- [x] Define the native output contract for `answers`, `selected_options`, `question_statuses`
- [x] Build synthetic unit tests that cover all question statuses
- [x] Build a small labeled real-image ORM case set
- [x] Verify Python ORM output against expected labels on that set
- [x] Implement native bubble scoring parity
- [x] Implement native question resolution parity
- [x] Run Python-vs-native parity on the labeled ORM case set
- [ ] Confirm downstream can consume the adapter-shaped output without reading Python-only helper fields

## Current Progress Notes

- Synthetic parity is passing for:
  - `single`
  - `multiple`
  - `invalid_multiple_on_single`
  - `uncertain`
- Best-path real-image parity is green for `answers` and `score`.
- Mixed `selection_mode` real-image smoke parity is green on `samples/1photo5.jpg`.
- Labeled real-image ORM case set is now checked in at `native_core/tests/data/orm_real_cases.json`.
- Current labeled real-image verification is green at `14/14` for both Python and native on the DLL test build.
- Best-path real-image parity is green for `answers`, `selected_options`, and `question_statuses` on the tuned DLL build.

## Exit Criteria

- Native output matches Python ORM semantics on the synthetic suite
- Native output matches Python ORM semantics on the labeled real-image suite
- Adapter contract is stable enough that BE integration does not depend on Python internals
