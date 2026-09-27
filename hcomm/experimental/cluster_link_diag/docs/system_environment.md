# 项目构建

## 安装系统环境和三方库

1. 指定目录。

   ```bash
   export THIRDLIB_ROOT=/usr/local/third_lib
   export ASCEND_HOME_PATH=/usr/local/Ascend
   export ASCEND_CANN_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
   ```

2. 安装系统环境。

   Ubuntu：

   ```bash
   sudo ./install.sh --install-system-packages --system ubuntu
   ```

   EulerOS/openEuler：

   ```bash
   sudo ./install.sh --install-system-packages --system euleros
   ```

3. 安装第三方库。

   C/C++：CLI11、nlohmann_json、Eigen、rpclib、可选 fmt/spdlog。

   Python venv：paramiko、scp、matplotlib。

   ```bash
   sudo -E ./install.sh --prefix "$THIRDLIB_ROOT" --jobs "$(nproc)" --strict-prereq
   ```

## 编译

```bash
source "$THIRDLIB_ROOT/share/disp_probe/third_party/env.sh"

cmake -S . -B build \
  -DTHIRDLIB_ROOT="$THIRDLIB_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"
```

## 产物

```bash
./build/rpc_host
./build/probe_topo
./build/probe_controller
./build/controller
```

测试和辅助可执行文件统一输出到：

```bash
./build/internal/
```
