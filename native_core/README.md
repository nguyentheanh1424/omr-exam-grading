# Native Core

This folder contains the native C++ core and exported C API for OMR grading and warp.

For the current `main` branch documentation, see:

- [`../docs/README.md`](../docs/README.md)
- [`../docs/native-c-api.md`](../docs/native-c-api.md)
- [`../docs/pipeline-guide.md`](../docs/pipeline-guide.md)

## Build

```powershell
cmake -S native_core -B build/native_core
cmake --build build/native_core --config Release
```

## Run tests

```powershell
cmake -S native_core -B build/native_core -G Ninja
cmake --build build/native_core
ctest --test-dir build/native_core --output-on-failure
python -m native_core.tests.parity_grading
```

## Integration notes

- External app or backend code should normalize data before calling the native API.
- Native core should stay free of direct JSON parsing.
- Result buffers returned by `omr_process` must be released with `omr_free_result`.
