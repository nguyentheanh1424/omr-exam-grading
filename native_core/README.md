# Native Core

This folder contains the native C++ core and exported C API for OMR grading.

## Current status
- Implemented:
  - C API lifecycle (`create/process/free/destroy`)
  - canonical input contract
  - strict validation
  - aligned-input grading core (bubble-score + threshold decision)
  - global homography warp
  - marker detect v1 from raw image (geometry-based, no OpenCV dependency)
  - optional global IDW refinement after homography (`warp_params.use_global_idw`)
  - optional local region refine + patch IDW (`warp_params.use_region_refine`)
- Not implemented in this cut:
  - full AprilTag ID decoding parity with Python `aruco` detector

## Build
```powershell
cmake -S native_core -B build/native_core
cmake --build build/native_core --config Release
```

## Run minimal example
```powershell
.\build\native_core\Release\omr_minimal_example.exe
```

## Run tests
```powershell
# configure/build (inside VS dev command prompt)
cmake -S native_core -B build/native_core -G Ninja
cmake --build build/native_core

# C++ unit tests
ctest --test-dir build/native_core --output-on-failure

# parity test (C++ vs Python reference)
python native_core/tests/parity_grading.py
```

## Integration notes
- App/web adapter must normalize external data into structs declared in `include/omr_api.h`.
- No JSON parsing should be added to the native core.
- Result buffers are owned by core and must be released with `omr_free_result`.
