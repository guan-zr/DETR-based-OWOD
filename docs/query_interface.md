# PROB+OWOBJ Decoder Query 的实际接口

依据当前B0实际运行文件`experiments/B0_PROB_OWOBJ/checkpoints/task1/config.json`及本地模型源码。
R1 worktree基于`ce7bc6b212889586ed596720b0e3a62a90f75606`；没有修改模型forward。

## 实际结构

| 项目 | 当前B0实际值 |
| --- | --- |
| 模型 | `models/prob_deformable_sketch_detr.py::DeformableDETR`，model_type=sketch |
| Decoder | 6层，每层100个Query，256通道 |
| 分类输出 | 每层[B,100,81]，Known为0..19，未来类别20..79屏蔽，unknown=80 |
| Query初始化 | `nn.Embedding(100,512)`，拆成256维query_pos及256维初始tgt |
| two_stage / with_box_refine / aux_loss | False / False / True |

具体位置：

- `models/prob_deformable_sketch_detr.py::DeformableDETR.__init__` 创建query_embed和各预测头。
- `models/deformable_transformer.py::DeformableTransformer.forward` 的非two_stage分支将query_embed拆成query_pos和tgt、扩展到Batch，并由query_pos经reference_points线性层得到初始参考点。
- 当前Encoder memory不参与Query初始化，随后通过各层cross-attention参与Query更新。源码中的two_stage Encoder proposal初始化分支当前未启用。
- `DeformableTransformerDecoder.forward`逐层保存output，返回`hs:[6,B,100,256]`及inter_references。
- `DeformableTransformerDecoderLayer.forward_ffn`最后执行`norm3`。因此hs本来就是该层最后一个LayerNorm之后的Query。

## R1提取的三种表示

| 字段 | 实际含义 |
| --- | --- |
| raw | 原模型hs，即每层norm3之后的输出；没有新增归一化 |
| pre_layernorm | 每层norm3的真实输入，通过forward_pre_hook捕获；不是整个Decoder层的输入 |
| l2 | raw的逐Query L2归一化副本，未修改模型，不拟合GT或测试统计量 |

`r1/features.py::FrozenQueryExtractor`挂接decoder输出及每个norm3输入。
整个检测器requires_grad=False、eval；提取forward为no_grad。
R1会检查每层aux输出`penu_layer`与对应hs完全相等，并核对Query/框/分数维度。
没有某层自身输出时明确报错，不用最后层框替代。

层号对外为1基。默认选1/3/6，实际层数不同时按`sorted(set(min(i,L) for i in [1,3,6]))`调整。
显式保存实际层号，不能将实际第4层标为第6层。各层独立训练配置记录requested_layer与actual_layer。

## 各层预测关系

`prob_deformable_sketch_detr.py::DeformableDETR.forward`循环每个hs[lvl]：

1. `class_embed[lvl](hs[lvl])`产生分类logits。
2. `prob_obj_head[lvl](hs[lvl])`产生概率物体性距离。
3. `bbox_embed[lvl](hs[lvl])`加该层参考点后sigmoid得到该层框。
4. 最后层放入顶层pred_logits/pred_boxes/pred_obj/penu_layer，其余层放入aux_outputs，顺序对应。

所以每层有自己的框数值，形状[B,100,4]、归一化cxcywh。当前with_box_refine=False，
class/bbox/objectness头通过ModuleList重复引用同一个模块，**参数共享不表示各层框相同**。
Decoder内没有逐层bbox refinement，但外层仍对每层Query计算预测框。

R1逐图保存[L_saved,100,C]特征、[L_saved,100,4]框、分类、距离/概率、层号和Query索引。
框转成原图XYXY时按各自图像的原始W/H缩放；原尺寸、实际resize尺寸、padded尺寸一并保存。
Query slot本身没有图像Padding slot；Padding由原检测器的Encoder/cross-attention mask处理。

## PROB与OWOBJ的物体性和评分

- 原PROB：对Query做无affine BatchNorm标准化，平方范数为近似对角Mahalanobis距离d。
  原PROB的PostProcess使用`exp(-temperature*d)*sigmoid(class_logits)`，再取全局Top100。
- OWOBJ集成：仍逐层计算距离；训练时还生成扰动先验及其BN统计。
  训练返回的pred_obj可为[B,2Q]（原始与扰动距离拼接），这不表示Decoder有200个Query。
  冻结eval时距离是[B,Q]。
- OWOBJ训练的unknown软目标为`exp(-(obj_temp/hidden_dim)*d)`；当前temperature=1.3/256。
  物体性及相关辅助损失应用最后层和各辅助层，而非只用最后层训练。
- 最终普通推理只返回并筛选最后层。公开OWOBJ `PostProcess.forward`里的Gaussian相乘被注释，
  实际是有效类别logits的sigmoid，在全部Query/class对中取Top100。
  没有新增NMS或R1阈值；一个Query可能以多个类别重复进入Top100。
- R1分别记录Gaussian概率及classifier_unknown sigmoid，不能把前者直接当成最终检测分数。

最终层selected_unique表示真实Top100 pair中出现过的去重Query集合。
中间层selected_unique是用同一规则对该层logits做的**离线模拟集合**，不是原模型实际中间层输出。
每个pair的Query ID、类别、分数仍原样保存，去重只用于Query统计防止重复计数。

## 已验证与待验证

CPU测试已验证6层合成结构、1/3/6各自的feature/box、norm3前后、Batch与原图缩放、
冻结参数/BN缓冲区、缓存往返及错误索引拒绝。
官方PostProcess也在合成Tensor上实际运行，最终框/类别/分数与R1映射完全相等。
这不等于真实Deformable DETR GPU前向已验证；须等B0完成后运行`python -m r1.validate_real`。
