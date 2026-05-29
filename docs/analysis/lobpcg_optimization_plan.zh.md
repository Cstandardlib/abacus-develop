# LOBPCG 性能优化与验证记录

## 目标

按“主要瓶颈到次要瓶颈”的顺序诊断并优化 LOBPCG 性能问题。每个重要节点记录可复核的正确性、收敛性和性能结果，避免用放松收敛判据掩盖算法热点。

## 执行原则

- 基线使用 `lobpcg_tol_scale = 0.2`，不把临时 `10.0` 作为正式基线。
- 构建使用 `/mnt/data/cn/buniverse.sh --no-cuda --test --release --no-install` 从头生成 `build_release_cpu_test`。
- 每轮只处理一个瓶颈；先做快速正确性，再做 Si2/4GaAs 分层验证。
- 重要节点先记录结果和风险判断，提交 commit 前等待审核。

## 待办

1. 建立 `lobpcg_tol_scale = 0.2` 的干净基线。
2. 优先减少 `iter_update_x` 中完整 `H*X` 重算。
3. 简化 `ortho_against_y` 中对参考子空间的重复处理。
4. 评估 `nmax` 策略。
5. 在结构性优化后重新比较 tolerance scale。

## 运行环境

- 日期：2026-05-29
- 工作目录：`/mnt/data/cn/abacus-develop`
- 构建目标：CPU Release with tests
- 构建目录：`build_release_cpu_test`

## 基线准备

- 已恢复 `source/source_hsolver/hsolver_pw.cpp` 中 `lobpcg_tol_scale = 0.2`。
- 当前只计划修改 LOBPCG 相关文件和本文档；工作树中的其它未跟踪文件视为既有本地产物。

## 结果记录

后续基线和优化结果追加在本节。

### 2026-05-29 基线：`lobpcg_tol_scale = 0.2`

构建命令：

```bash
/mnt/data/cn/buniverse.sh --no-cuda --test --release --no-install
```

构建结果：成功，生成 `build_release_cpu_test/abacus`。链接阶段出现 `libmpi.so.12` 与 `libmpi.so.40` 可能冲突警告，但构建未失败。

单元测试：

```bash
ctest --test-dir build_release_cpu_test -R "MODULE_HSOLVER_lobpcg|MODULE_HSOLVER_lobpcg_simple|MODULE_HSOLVER_pw" --output-on-failure
```

- sandbox 内首次运行 `MODULE_HSOLVER_lobpcg` 因 OpenMPI socket 权限失败；提权后 MPI 初始化正常。
- `MODULE_HSOLVER_lobpcg_simple`：通过。
- `MODULE_HSOLVER_pw`：通过。
- `MODULE_HSOLVER_lobpcg`：基线失败，失败用例为 `DiagoLOBPCGTest.readH`，500 次未收敛，最低 4 个本征值相对 LAPACK 超过 `0.1` 阈值。其它 5 个 LOBPCG 测试通过。该失败发生在本轮优化前，作为既有风险记录。

算例运行目录：

- Si2 np=1：`/tmp/abacus_lobpcg_si2_np1.5cKOe2`
- Si2 np=28：`/tmp/abacus_lobpcg_si2_np28.aMojzH`
- 4GaAs np=1：`/tmp/abacus_lobpcg_4gaas_np1.ucxwjl`
- 4GaAs np=28：`/tmp/abacus_lobpcg_4gaas_np28.I102B5`

| Case | np | SCF steps | Final energy (eV) | Total (s) | `Diago_LOBPCG diag` | `Operator hPsi` | `hPsi` calls | `iter_hpsi` | `iter_update_x` | `ortho_against_y` | `rayleigh_ritz` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Si2 | 1 | 5 | -215.5056984551635 | 13 | 11.91 | 9.87 | 582 | 3.62 | 4.81 | 0.98 | 0.36 |
| Si2 | 28 | 5 | -215.5056984542014 | 3 | 2.55 | 1.77 | 574 | 0.65 | 0.87 | 0.27 | 0.34 |
| 4GaAs | 1 | 6 | -7836.7171487724936 | 908 | 877.59 | 513.26 | 2644 | 183.22 | 338.40 | 213.92 | 35.83 |
| 4GaAs | 28 | 6 | -7836.7171487741671 | 97 | 95.12 | 46.54 | 2626 | 17.21 | 29.20 | 15.03 | 19.56 |

观察：

- Si2 np=1/28 与 4GaAs np=1/28 的最终能量在并行度间一致。
- 4GaAs 上 `iter_update_x` 是主要热点：np=1 为 338.40 s，np=28 为 29.20 s。
- `iter_update_x` 当前包含 `hx_new = hspace * h_red` 后的 `ortho(x_new)` 和完整 `hpsi_func(x_new)` 重算；这是第一轮优化目标。

### 2026-05-29 优化 1：避免 `iter_update_x` 无条件完整 `H*X` 重算

改动内容：

- `iter_update_x` 仍用 `hx_new = hspace * h_red` 形成 Ritz 向量对应的 `H*X`。
- 对 CPU 路径新增全局正交性检查 `X^H X ~= I`，检查通过时保留线性组合得到的 `hx_new`。
- 若是广义本征值问题、GPU/ROCm 路径或正交性检查失败，则回退到旧路径：重新 `ortho(x_new)` 并调用完整 `hpsi_func(x_new)`。
- GPU/ROCm 当前保持保守回退，不改变设备路径行为。

构建与单元测试：

```bash
/mnt/data/cn/buniverse.sh --no-cuda --test --release --no-install
ctest --test-dir build_release_cpu_test -R "MODULE_HSOLVER_lobpcg|MODULE_HSOLVER_lobpcg_simple|MODULE_HSOLVER_pw" --output-on-failure
```

- 构建成功，生成 `build_release_cpu_test/abacus`。链接阶段仍有 `libmpi.so.12` 与 `libmpi.so.40` 可能冲突警告，与基线一致。
- `MODULE_HSOLVER_lobpcg_simple`：通过。
- `MODULE_HSOLVER_pw`：通过。
- `MODULE_HSOLVER_lobpcg`：仍失败于既有 `DiagoLOBPCGTest.readH`；其它 5 个 LOBPCG 测试通过。失败特征与基线一致，仍是 `ortho_against_y` 反复未达 tolerance 后 500 次未收敛。

算例运行目录：

- Si2 np=1（重建后快速复核）：`/tmp/abacus_lobpcg_opt1_post_si2_np1.8BSpWG`
- Si2 np=28（重建后快速复核）：`/tmp/abacus_lobpcg_opt1_post_si2_np28.a9Hxi3`
- 4GaAs np=1：`/tmp/abacus_lobpcg_opt1_4gaas_np1.XGhsT3`
- 4GaAs np=28：`/tmp/abacus_lobpcg_opt1_4gaas_np28.grQd4C`

优化后结果：

| Case | np | SCF steps | Final energy (eV) | Total (s) | `Diago_LOBPCG diag` | `Operator hPsi` | `hPsi` calls | `iter_hpsi` | `iter_update_x` | `ortho_against_y` | `rayleigh_ritz` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Si2 | 1 | 5 | -215.505698454864 | 10 | 8.64 | 6.46 | 340 | 4.30 | 0.25 | 1.11 | 0.41 |
| Si2 | 28 | 5 | -215.5056984545857 | 3 | 1.96 | 1.14 | 337 | 0.77 | 0.05 | 0.32 | 0.34 |
| 4GaAs | 1 | 6 | -7836.7171487714604 | 594 | 563.02 | 244.42 | 1452 | 181.57 | 27.44 | 213.26 | 35.55 |
| 4GaAs | 28 | 6 | -7836.7171487745436 | 73 | 70.29 | 23.12 | 1467 | 17.28 | 4.20 | 15.36 | 19.62 |

相对基线变化：

| Case | np | Total | `Diago_LOBPCG diag` | `Operator hPsi` calls | `Operator hPsi` time | `iter_update_x` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Si2 | 1 | 13 -> 10 s | 11.91 -> 8.64 s | 582 -> 340 | 9.87 -> 6.46 s | 4.81 -> 0.25 s |
| Si2 | 28 | 3 -> 3 s | 2.55 -> 1.96 s | 574 -> 337 | 1.77 -> 1.14 s | 0.87 -> 0.05 s |
| 4GaAs | 1 | 908 -> 594 s | 877.59 -> 563.02 s | 2644 -> 1452 | 513.26 -> 244.42 s | 338.40 -> 27.44 s |
| 4GaAs | 28 | 97 -> 73 s | 95.12 -> 70.29 s | 2626 -> 1467 | 46.54 -> 23.12 s | 29.20 -> 4.20 s |

观察与风险：

- 正确性：Si2 与 4GaAs 的 np=1/28 最终能量继续在基线范围内一致；SCF 步数不变。
- 性能：第一热点 `iter_update_x` 已被显著压低。4GaAs np=1 从 338.40 s 降至 27.44 s，np=28 从 29.20 s 降至 4.20 s。
- `hPsi` 调用数量约减半，说明避免了主循环每次迭代末尾的完整 `H*X` 刷新。
- 新的主要热点变为 `ortho_against_y` / `iter_update_space`，符合下一优先级。
- 当前正交性检查每次迭代额外做一次 `X^H X`；在 4GaAs 上代价远小于被移除的完整 `hpsi_func`，但后续可考虑把检查频率或阈值做成调试/保护路径。
- 本节点建议先审核；通过后再提交，不继续混入 `ortho_against_y` 优化。
