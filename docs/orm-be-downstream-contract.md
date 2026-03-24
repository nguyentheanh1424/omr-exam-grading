# ORM BE/Downstream Contract

## Scope

This contract defines:

- what `BE` should send into the ORM adapter
- what the Python adapter sends to native
- what downstream should read from the ORM result

Architecture:

- `BE -> Python adapter -> Native core`

Native core does not read BE JSON directly.

## Official Integration Flow

Official flow:

- `BE -> Python adapter -> Native raw warp + Native ORM/Grading`

Fallback/debug flow:

- `BE -> Python adapter -> Python warp/prep -> Native ORM/Grading`

The fallback path is kept for verification and troubleshooting, but the official product path is the native raw-warp path.

## Contract Boundary

### BE owns

- exam/business config
- bubble layout
- `selection_mode` per question
- answer key
- threshold overrides if needed

### Python adapter owns

- JSON parsing
- validation
- normalization
- mapping to native structs
- deriving helper fields for downstream

### Native core owns

- bubble scoring
- question resolution
- returning core ORM fields

## BE Input Contract

### Required fields

For each bubble ROI:

- `cx`
- `cy`
- `r`
- `question`
- `option`
- `selection_mode`

For the whole request:

- `answer_key`
- optionally threshold config:
  - `abs_th`
  - `rel_th`
  - `auto_threshold`

### Allowed selection_mode values

- `single`
- `multiple`

All bubbles belonging to the same question must use one consistent `selection_mode`.

### Indexing

BE-facing question numbering stays `1`-based.

Option numbering stays `0`-based.

Python adapter is responsible for converting question indices to native `0`-based space.

## Recommended BE Payload Shape

```json
{
  "bubble_layout": [
    {
      "cx": 803,
      "cy": 1198,
      "r": 22,
      "question": 1,
      "option": 0,
      "selection_mode": "single"
    }
  ],
  "answer_key": [0, 2, 1, -1],
  "thresholds": {
    "abs_th": 0.2,
    "rel_th": 0.0555,
    "auto_threshold": false
  }
}
```

Notes:

- `answer_key[i] = -1` means that question is excluded from grading.
- If `selection_mode` is omitted by BE, Python may currently default it to `single`, but BE should send it explicitly for contract clarity.

## Adapter Validation Rules

Python adapter must reject:

- missing ROI fields
- invalid `selection_mode`
- mixed `selection_mode` inside one question
- non-contiguous question numbering in BE space
- inconsistent option sets across questions
- answer key length mismatch

## Native v1 Output Contract

Native returns only core fields:

- `answers`
- `selected_options`
- `question_statuses`

### answers

- valid single answer -> selected option index
- `blank` -> `-1`
- `multiple` -> `-1`
- `invalid_multiple_on_single` -> `-1`
- `uncertain` -> `-1`

### selected_options

Per-question list of bubbles considered filled.

This is the primary field for preserving multiple-mark information.

### question_statuses

Allowed values:

- `blank`
- `single`
- `multiple`
- `invalid_multiple_on_single`
- `uncertain`

## Question Resolution Rules

### single mode

- 0 filled -> `blank`
- 1 confident filled -> `single`
- 2+ filled -> `invalid_multiple_on_single`
- borderline ambiguity -> `uncertain`

### multiple mode

- 0 filled -> `blank`
- 1 filled -> `single`
- 2+ strong filled -> `multiple`
- recovered borderline second bubble -> `uncertain`

## Downstream Contract

Downstream should read:

- `question_statuses`
- `selected_options`
- `answers` only for backward-compatible grading flows

Downstream should not infer question semantics from `answers` alone.

### Recommended downstream interpretation

- `single` -> consume `answers[i]`
- `multiple` -> consume `selected_options[i]`
- `invalid_multiple_on_single` -> treat as invalid response
- `uncertain` -> treat as unresolved/manual-review candidate
- `blank` -> unanswered

## Adapter-Shaped Output

Python adapter may derive helper fields such as:

- `multiple_questions`
- `invalid_multiple_on_single_questions`
- `blank_questions`
- `single_questions`
- `uncertain_questions`
- `status_counts`

These are convenience fields for UI/API consumers and are not part of the native core contract.

## Recommended Response Shape For App/BE

```json
{
  "answers": [0, -1, -1, 2],
  "selected_options": [[0], [], [1, 3], [2]],
  "question_statuses": ["single", "blank", "multiple", "single"],
  "question_selection_modes": ["single", "single", "multiple", "single"],
  "multiple_questions": [3],
  "invalid_multiple_on_single_questions": [],
  "uncertain_questions": [],
  "blank_questions": [2],
  "single_questions": [1, 4]
}
```

## Integration Rule

If there is any conflict between:

- native raw core output
- adapter helper fields
- UI-specific convenience fields

then the source of truth order is:

1. `question_statuses`
2. `selected_options`
3. `answers`
