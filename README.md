# terrain-splat

## 题面

地形换成多层混合材质后，远处发亮、接缝处一条黑线：权重不归一，四层叠加后亮度翻倍；全零权重除零，输出 NaN；平铺尺寸向下取整到 0，采样除零；UV 不 wrap，边界处出现一条黑线；坡度混合只用高度，陡坡也长草；高度权重硬编码阈值，不同地形尺度失效；mip 偏移不随距离变化，远处闪烁；跨 tile UV 不平移，接缝错位半个 tile；超过层数上限直接截断，剩余权重不归零；批量遇非法 tile 只跳过，输出缺块。terrain/splat.py 是核心，tests/test_splat.py 15个红。要修的是：按 8 个函数的 docstring 契约实现权重归一、tile 尺寸、UV wrap、坡度/高度混合、mip 偏移、跨 tile 缝合、层数上限与批量回滚，让 15 个测试全部通过。

## 运行

```powershell
python -m pytest tests -q
```

初始失败环境应恰好输出 `15 failed`，且不能出现 collection、import 或 syntax error。实现完成后目标是 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 文件结构

```text
terrain-splat/
  README.md  conftest.py  .gitignore
  terrain/__init__.py  terrain/splat.py
  tests/test_splat.py
```

`terrain/splat.py` 中的 8 个公开函数当前全部是 `raise NotImplementedError` 的桩；权重范围、UV wrap、单位与 1e-6 容差写在各函数的 docstring 中，异常统一使用 `SplatError`。

## 数据形状

测试中的图层用内联字典构造，基础字段为 `{"id": str, "weight": float, "uv_scale": float, "tile": int}`；`blend` 额外读取 `"color": (r, g, b)`，三个分量均在 0..1。批量接口的每个 tile 为 `{"id": str, "layers": [...], "uv": (u, v)}`。

## 约束

- `tests/test_splat.py` 不可修改；只允许实现 `terrain/splat.py` 中的桩函数。
- 仅使用标准库（`math`）与 `pytest`；离线、无随机、不联网。
- 15 个测试在初始仓库中必须全部失败，且失败类型来自桩抛出的 `NotImplementedError`，不能是 collection、import 或 syntax error。
