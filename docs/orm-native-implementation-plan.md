# ORM Native Implementation Plan

## Objective

Implement ORM native support without coupling native code to BE JSON format.

Target architecture:

- `BE -> Python adapter -> Native core`

Deployment decision now locked:

- official flow uses native raw warp plus native ORM/grading
- Python warp/prep remains a fallback verification path

## Phase 1

Build native parity for core ORM logic only.

### Scope

- bubble scoring
- question resolution
- output serialization for:
  - `answers`
  - `selected_options`
  - `question_statuses`

### Out of Scope

- BE JSON parsing in native
- Python-only derived fields
- UI/helper reports

## Work Breakdown

### 1. Adapter Contract

- Add adapter-facing native input fields for bubble ROIs
- Include `selection_mode` per question or per ROI
- Include threshold config
- Validate question consistency in Python before native call

### 2. Native Bubble Scoring

- Reproduce current Python bubble scoring behavior
- Verify patch extraction and annulus regions match Python
- Add focused unit tests for score parity

### 3. Native Question Resolution

- Implement current Python v1 semantics for:
  - `single`
  - `multiple`
  - `invalid_multiple_on_single`
  - `blank`
  - `uncertain`
- Keep status values identical to Python

### 4. Native Output Shape

- Return raw `answers`
- Return `selected_options`
- Return `question_statuses`

### 5. Python Integration

- Update Python adapter to call native ORM path
- Keep Python-side derived output fields unchanged
- Preserve backward-compatible `answers` behavior

## Verification

### Synthetic

- unit tests for all statuses
- unit tests for threshold edge cases
- unit tests for `selection_mode` consistency

### Real Cases

- labeled small ORM case set
- compare Python vs native on:
  - `answers`
  - `selected_options`
  - `question_statuses`

Current baseline:

- `native_core/tests/data/orm_real_cases.json`
- `native_core/tests/parity_orm_real_cases.py`

## Suggested Execution Order

1. Freeze adapter input/output contract
2. Port bubble scoring
3. Port question resolution
4. Add parity tests
5. Integrate into Python adapter
6. Run real-case verification
