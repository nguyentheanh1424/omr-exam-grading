# ORM Native v1 Contract

## Architecture

- `BE -> Python adapter -> Native core`

Official runtime path:

- `BE -> Python adapter -> Native raw warp + Native ORM/Grading`

Fallback/debug path:

- `BE -> Python adapter -> Python warp/prep + Native ORM/Grading`

## Input Ownership

BE owns business config.

Python adapter owns:

- parsing
- validation
- normalization
- mapping to native structs

Native core owns:

- bubble scoring
- question resolution

## Required Input Data

For each bubble ROI:

- `cx`
- `cy`
- `r`
- `question`
- `option`
- `selection_mode`

Question numbering at the BE/Python level remains `1`-based if that is the current app contract.
Python adapter may normalize indices before sending them to native.

## selection_mode Values

Allowed values:

- `single`
- `multiple`

All bubbles belonging to the same question must resolve to one consistent `selection_mode`.

## Threshold Input

Adapter may send:

- `abs_th`
- `rel_th`
- auto-threshold flags if needed later

For v1, native should follow the same semantics as current Python ORM:

- `abs_th` is the main fill threshold
- `rel_th` is used for single-answer confidence

## Output Fields

Native returns:

- `answers`
- `selected_options`
- `question_statuses`

### answers

- For valid single-answer questions, `answers[i]` is the selected option index
- For `blank`, `multiple`, `invalid_multiple_on_single`, and `uncertain`, `answers[i] = -1`

### selected_options

- List of selected option indices for each question
- This is the main field used to preserve multiple-mark information

### question_statuses

Allowed values for v1:

- `blank`
- `single`
- `multiple`
- `invalid_multiple_on_single`
- `uncertain`

## Question Resolution Semantics

### single mode

- 0 filled bubbles -> `blank`
- 1 confident filled bubble -> `single`
- 2+ filled bubbles -> `invalid_multiple_on_single`
- borderline ambiguity -> `uncertain`

### multiple mode

- 0 filled bubbles -> `blank`
- 1 filled bubble -> `single`
- 2+ strong filled bubbles -> `multiple`
- recovered borderline second bubble -> `uncertain`

## Python Adapter Responsibilities After Native Call

Python adapter may derive:

- per-status question lists
- status counts
- grading summaries
- UI-friendly fields

Those derived fields are outside the native v1 contract.
