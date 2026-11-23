# 实际模型接口与空间坐标

本说明基于本地官方 PROB 集成模型 `models/prob_deformable_sketch_detr.py`，
原始 OWOBJ 官方 Commit `f5c583e39593168e2313c149ba69801d79619f42`，
R0 worktree 基于 B0 Commit `ce7bc6b212889586ed596720b0e3a62a90f75606`。

## 源码位置

| 接口 | 实际源码/位置 | 输出与含义 |
| --- | --- | --- |
| Backbone | `models/backbone.py:75`，`BackboneBase.body` 返回 layer2/3/4 | 3级，通道512/1024/2048，stride8/16/32 |
| Backbone forward 返回值 | `models/backbone.py:86`，外层 `Joiner.forward` | 每级 NestedTensor(tensors,mask)，另返回位置编码 |
| S1 | `models/prob_deformable_sketch_detr.py:264` 到调用 transformer 的288行 | `srcs`：前三层经 Conv1×1 + GroupNorm 投影，第四层由最后一级 Backbone 经 Conv3×3 stride2 + GroupNorm 产生；均256通道 |
| S1 只读挂点 | `model.transformer.register_forward_pre_hook` | inputs[0]=srcs，inputs[1]=masks，inputs[2]=pos_embeds；读取尚未加入位置/层级编码的投影特征 |
| Encoder flatten | `models/deformable_transformer.py:139` | 每级 BCHW → B(HW)C，按尺度顺序拼接；每级内部W维变化最快 |
| S2 | `models/deformable_transformer.py:159` | `memory = self.encoder(...)`，形状[B,sum(H_l×W_l),256] |
| S2 只读挂点 | `model.transformer.encoder.register_forward_hook` | 根据S1实际H/W逐段split，再transpose(1,2).reshape(B,C,H,W) |
| Query预测 | 模型输出 `pred_boxes` / `pred_logits` | 最后一层[B,100,4]归一化cxcywh / [B,100,81]；R0只将框作为ROI，不重建Query |

两条路线输出4级：`[B,256,H_l,W_l]`，名义stride `[8,16,32,64]`。
H/W以实际hook读到的形状为准，CNN有ceil/取整，不能用固定图片尺寸猜测。
例如padded输入800×1344时四级为100×168、50×84、25×42、13×21。

本次CPU合成结构验证使用8通道假检测器，验证真实hook边界的调用约定、token顺序和冻结行为；
另用真实256通道AE单独验证尺寸和反向。
本次还在CPU成功构建官方39,964,695参数检测器，确认3级Backbone、4级投影、6层Encoder，
没有执行官方检测器forward。官方Deformable Attention只有CUDA前向，真实完整输出一致性
由 `r0.validate_detector` 在B0结束后验证。

## Padding与冻结

- 原模型mask为True表示padding，按官方实际mask逐级读取，不自行重建近似边界。
- `FrozenSpatialExtractor` 对整个检测器 `requires_grad_(False)`、`eval()`，forward为`no_grad()`。
  即使外层调用train，检测器也保持eval；防止BN统计量或dropout变化。
- 捕获值detach；padding清零使用out-of-place操作，不更改原检测器输入、返回值或中间张量。
- AE再次对目标detach；随机mask只落在有效空间位置，同位置全部通道共同掩码。
- Padding不参与MSE、不参与ROI均值；图像ID、原尺寸、resize尺寸、padded尺寸随SpatialBatch保存。

## 坐标约定

所有评价候选首先位于原图连续XYXY坐标。
XML的xmin/ymin减1，xmax/ymax不变，与PROB加载器一致。
原图框先乘实际resize后的独立x/y比例，再除该级**名义stride**进入ROIAlign。
不以H_feature/H_padded代替stride，避免奇数padding产生比例偏差。

ROIAlign使用aligned=True、固定7×7 bins、sampling_ratio=2；支持图与误差乘积共同采样：
`region_mean = sum(ROIAlign(E*support)) / sum(ROIAlign(support))`。
这会排除padding及掩码专用误差中的无效位置。
非有限/退化框无效；框裁剪到图像边界；resize后小于1像素无效；
小于一个feature cell的框保留并打`small_feature_roi`标记。
区域有效支持不足50%时主分数无效并保存原因所需的valid_fraction；
masked-only辅助分数只要求支持非零，空mask返回缺失值而不是误差0。

## 扩展边界

`mean_region_error(error,support,boxes,geometry,stride) -> RegionScores` 是独立聚合接口。
后续可在regions模块增加AREA聚合策略而不改检测器或FMR目标；当前只实现普通均值。
没有Weibull、融合评分、重建Query或修改OWOBJ损失。
