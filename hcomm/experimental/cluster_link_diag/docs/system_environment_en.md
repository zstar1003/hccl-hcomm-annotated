# Project Build

## Install System Environment and Third-Party Libraries

1. Specify directories.

   ```bash
   export THIRDLIB_ROOT=/usr/local/third_lib
   export ASCEND_HOME_PATH=/usr/local/Ascend
   export ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
   ```

2. Install the system environment.

   Ubuntu:

   ```bash
   sudo ./install.sh --install-system-packages --system ubuntu
   ```

   EulerOS/openEuler:

   ```bash
   sudo ./install.sh --install-system-packages --system euleros
   ```

3. Install third-party libraries.

   C/C++: CLI11, nlohmann_json, Eigen, rpclib, and optional fmt/spdlog.

   Python venv: paramiko, scp, matplotlib.

   ```bash
   sudo -E ./install.sh --prefix "$THIRDLIB_ROOT" --jobs "$(nproc)" --strict-prereq
   ```

## Compilation

```bash
source "$THIRDLIB_ROOT/share/disp_probe/third_party/env.sh"

cmake -S . -B build \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
```

## Artifacts

```bash
./build/rpc_host
./build/probe_topo
./build/probe_controller
./build/controller
```

Test and auxiliary executables are output to:

```bash
./build/internal/
```
