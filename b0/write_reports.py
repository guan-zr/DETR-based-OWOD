"""Generate honest progress/final reports from on-disk measured artifacts only."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from b0.common import ARTIFACTS, REPO


def read(relative,default=None):
    path=ARTIFACTS/relative
    return json.loads(path.read_text()) if path.exists() else default


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    data=read('reports/data_audit.json',{})
    smoke=read('reports/smoke.json')
    stress=read('reports/memory_stress.json')
    final=read('query_analysis/final/summary.json')
    status=read('reports/train_status.json',{})
    source=read('reports/source_verification.json',{})
    env=read('reports/environment.json',{})
    config=json.loads((REPO/'b0/config.json').read_text())
    now=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    free=shutil.disk_usage(ARTIFACTS).free/1024**3
    smoke_checkpoint=smoke.get('checkpoint','待运行') if smoke else '待运行'
    base=f'''更新时间：{now}（Asia/Shanghai）

实验范围仅 B0、M-OWODB Task 1。未开发重建网络、重建损失或新评分融合。

## 环境与路径

- 执行位置：AutoDL Ubuntu 22.04 Linux 容器；未访问或修改 Windows 本地文件。
- 主项目：`{REPO}`；原始 PROB：`/root/autodl-tmp/PROB-main`（文件校验清单在 reports/prob_original_sha256.json）。
- 实验产物：`{ARTIFACTS}`。
- Conda 环境：`{ARTIFACTS}/env`，从已有 base 离线克隆后只在此环境安装依赖。
- GPU：RTX 4090 24GB，驱动 580.105.08；驱动显示 CUDA 13.0，实际 Toolkit/nvcc 与 PyTorch runtime 使用 CUDA 11.8。
- Python/PyTorch/TorchVision：{env.get('python','待记录')} / {env.get('torch','待记录')} / {env.get('torchvision','待记录')}。
- 实际版本：reports/environment.json、reports/pip_freeze.txt、reports/conda_environment.yml；编译日志 logs/cuda_build.log。
- COCO：`/root/autodl-tmp/datasets/COCO2017`；VOC：`/root/autodl-tmp/datasets/VOC/VOC2007`、`VOC2012`。
- 统一 OWOD：`/root/autodl-tmp/datasets/OWOD_B0`；图片全部为指向已有原图的软链接，VOC XML 也为软链接。
- 数据盘实际 207GB，初始剩余约 167GB；报告生成时剩余 {free:.2f} GiB。未清理原有回收站或其他项目。

## 官方实现、协议及差异

- [PROB 官方仓库](https://github.com/orrzohar/PROB)，对应远端 Commit `{source.get('prob',{}).get('official_commit','待核实')}`。
- [OWOBJ 官方仓库](https://github.com/AI4Math-ShanZhang/OWOBJ)，对应远端 Commit `{source.get('owobj',{}).get('official_commit','待核实')}`。抽查 README、模型、入口、数据加载、训练配置、评估器的 Git blob 哈希与现有源码一致，详见 source_verification.json。
- 现有目录没有 .git；先将原始 OWOBJ 快照提交到 `vendor-owobj`（`ca911786051508f41ed59f6873385267af43acb7`），再建立 `{branch}`。本地导入 Commit 不冒充上游 Commit。
- 直接使用官方集成的 `models/prob_deformable_sketch_detr.py`（`model_type=sketch`），保留 Deformable DETR、分类和定位。未再叠加独立 PROB objectness 损失。
- 概率物体性：所有 query 经 BatchNorm 估计均值与对角方差，匹配已知 GT 的 query 使用 Mahalanobis 距离损失；训练时额外生成随机扰动先验并计算 KL；未匹配 query 的 unknown 分类软标签由 `exp(-1.3/256 * distance)` 给出；加入能量间隔损失。
- **推理按公开代码**：对有效已知类和 unknown 的分类 logit 使用 sigmoid，在所有 query/class 对中取 Top100；不额外乘 Gaussian objectness，不另设 NMS 或新阈值。Query TopK 同时报告 latent objectness 与 classifier unknown sigmoid 两种排名。
- 官方名为 M 的训练脚本实为 OWDETR 的 19 类设置，且 `t1_train/test` 划分名称不匹配；采用 PROB 原始 M 配置和 OWOBJ 官方 M 评估配置一致的 **TOWOD、20 类**。原脚本保留。
- 官方加载器硬编码 `/g/data/...` 的 HDF5/COCO 路径，只能处理 COCO ID；恢复 PROB 的 JPEG/XML 后端，支持 M-OWODB 中的 VOC 图像，保留类别排序、转换与标签过滤逻辑。
- 修复评估调用不支持的 `coco=` 参数；移除前 100 batch 的调试中断，保证全量评估；NumPy `np.bool` 改为等价 `np.bool_`；避免测试重标记修改缓存中的原始类别。
- 官方入口 wandb 局部变量遮蔽修复，实验不连接 W&B、不上传数据。
- **论文/代码差异明确保留**：论文文字举例有 ResNet101，发布 PROB 集成配置为 DINO ResNet50；采用后者。发布代码分类采用 varifocal 实现，能量损失按每个 query 计算，KL 基于 `BN.running_mean/var.clone()`，这些统计量不带梯度，因此 KL 在发布代码中不直接反传。这些模型代码保持原样，不自行修正算法后宣称复现。
- 适配环境由官方 Torch1.12/cu113 改为 Torch2.1.2/cu118。单 GPU 使用梯度累积保持有效 batch20；BN 仍在每个物理 batch 内计算，不能声称与 4GPU 的 EMA 更新轨迹逐位一致。结论标记为**公开代码复现与工程适配**，不宣称与论文训练逐位严格等价。

## 数据完整性与 Unknown 定义

原始数据计数：COCO train2017=118287，val2017=5000；VOC2007=9963，VOC2012=17125。转换使用未修改的官方 `datasets/coco2voc.py`，保留整数取整、COCO crowd 处理和坐标约定。

完整数据审计：`reports/data_audit.json`。所有 Task 1 图片实际解码、XML 解析、尺寸与坐标检查，不通过路径错误跳过样本。

{table(['划分','条目数','唯一图片数','多余重复条目'],[[name,x['rows'],x['unique_images'],x['extra_rows']] for name,x in data.get('split_rows_and_unique',{}).items()])}

- 校验错误数：{len(data.get('errors',[]))}；训练/测试图片交叉：{data.get('train_test_overlap','待检查')}。
- 官方测试文件重复 ID 原样保留，官方评估会按原始行累计 GT 分母；**不私自去重以提高指标**。离线 Query 主要统计唯一图片中的物体，并明确区别官方分母。
- 官方测试逐行口径 Unknown 非 difficult GT：{data.get('splits',{}).get('owod_all_task_test',{}).get('unknown_non_difficult','待统计')}；这不是尚未计算的唯一 GT 数。
- 已知为 VOC20：aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor。
- Unknown 为测试 XML 中另外 60 个 COCO 类的已标注实例；未标注区域不视为 Unknown GT。difficult 不计召回分母。
- 训练加载器只保留当前已知类，训练脚本逐 batch 断言标签<20；不使用未知 GT 作训练监督或选择阈值、超参、最佳权重。

## 预训练与配置

- 官方 DINO ResNet50：`checkpoints/dino_resnet50_pretrain.pth`，来源 [DINO 官方权重](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth)。SHA256 与下载来源在 checkpoints/dino_provenance.json。
- DINO 为 ImageNet 自监督预训练，复用官方指定来源，见 [DINO官方说明](https://github.com/facebookresearch/dino)；未加载含 COCO 未知类框监督的通用检测器权重。不声称预训练图像不含未知语义，仅确认未使用其检测框标注。
- 未发现现有完整 Task1 权重；OWOBJ 官方 README 无可核实下载链接，不能把 PROB 权重冒充 PROB+OWOBJ 权重，故准备完整训练。
- 配置文件：`b0/config.json`，产物目录 configs/task1.json 为其链接。
- DINO R50、多尺度4层、Encoder6层、Decoder6层、hidden256、100 queries；图像采用官方 make_coco_transforms，测试短边800、长边上限1333，训练使用原始多尺度/裁剪增强。
- AdamW，lr=2e-4，backbone lr=2e-5，投影/参考点 lr multiplier=0.1，weight_decay=1e-4，clip norm=0.1；StepLR 在35 epoch 衰减；41完整 epoch，seed42。
- 物理 batch={config['batch_size']}、累积={config['accumulation_steps']}、有效 batch={config['batch_size']*config['accumulation_steps']}；FP32，未启用混合精度；每 epoch 使用16540图像，drop_last 与4GPU/每GPU5张的有效样本数相同。
- 损失：分类2、L1框5、GIoU2、objectness8e-4、KL0.1、energy1；obj_temp1.3、margin0.2，保留所有 Decoder 辅助损失。
- 不根据 Unknown 测试指标选择最佳 Checkpoint；官方无明确最佳权重选择规则，使用最终 epoch40。保留 rolling checkpoint、epoch0、LR下降前epoch34、最终epoch40，不大量重复保存。
- 磁盘预算：环境约5GB，共享 XML/软链接不足1GB，DINO约90MB，模型+Adam checkpoint 每份约0.5GB、4份约2GB，分析压缩样例与统计预计<0.1GB，日志预算<1GB；启动前强制剩余>=15GiB。

## CUDA 与 Smoke Test

- CUDA 扩展：原始 C++/CUDA 源码未修改，`TORCH_CUDA_ARCH_LIST=8.9`、CUDA11.8 编译成功。
- 原始 CUDA 单测：double/float 前向与PyTorch一致，数值梯度 D=30/32/64/71/1025通过；2048时完整 Jacobian 导致24GB测试OOM。保留失败日志。
- 适配检查使用原测试函数、fast_mode=True，允许 atomicAdd 重入末位漂移 nondet_tol=1e-10，全部7种通道含2048/3096通过；未放宽数值/解析梯度比较容差。日志 logs/cuda_test_fast_nondet.log。
- 数据加载、模型构建、Forward/Loss/Backward/Optimizer、少量训练、Checkpoint strict加载、官方子集评估：{'已通过，详见 reports/smoke.json' if smoke else '尚待完成，不能称为通过'}。
- 接口前后预测完全一致检查：{smoke.get('feature_hooks_bitwise_equal') if smoke else '尚待完成'}；特征形状记录在 smoke.json。
- 所有 Smoke 指标仅验证流程，不当作完整复现成绩。
- 显存边界：初始batch5真实40张Smoke通过，但1333×1333最大padding训练OOM（logs/memory_stress.log）。因此采用batch2×累积10，重新完整Smoke；最大padding测试结果：{stress if stress else '待运行'}。两版Smoke权重与日志均保留。

## 可复用接口和统计规则

- `b0/features.py::FeatureCapture` 用只读 hooks 读取 Backbone及mask、Transformer投影特征、Encoder序列、Decoder全部层query、各层框/logits/距离。启用方式：`with torch.no_grad(), FeatureCapture(model) as capture: outputs=model(samples)`。仅 eval 模式，不改变返回值。
- Backbone为3尺度[B,512/1024/2048,H,W]；投影为4尺度[B,256,H,W]；Encoder[B,sum(H_l*W_l),256]（按层和行优先展开）；Decoder[L,B,100,256]；各层框[L,B,100,4]。特征层索引/Query索引与预测对应。
- 分析在原图像素连续XYXY坐标计算IoU；GT xmin/ymin减1，xmax/ymax不变，与官方加载器保持一致；无score过滤的原始最后层100框用于候选覆盖率。每个GT独立取最大IoU，允许同一query覆盖多个GT，不等于一对一召回。
- TopK=10/50/100；latent为exp(-1.3/256*d)，另报classifier unknown sigmoid。Top100必须等于全部query覆盖率。
- Query互斥分组：对全部标注GT找最近IoU，>=0.5且GT非difficult才分Known/Unknown；与全部GT最大IoU<0.1才分Background，其余含difficult匹配均为Ambiguous并单独计数。并非“不匹配Known就是Background”；背景仅相对现有标注定义。
- A：官方unknown匹配未检出、但某个原始query IoU>=0.5；B：官方unknown未检出且所有query IoU<0.5。分别报告占全部Unknown和漏检Unknown的比例。A含错分类、排名、重复竞争等，不能全归因于低物体性。
- 最终检测使用官方评估器的三位score/一位框坐标格式化、IoU>0.5、一对一匹配，并校验离线TP数重放与官方U_R50一致；候选阈值为>=0.3/0.5/0.7，边界严格程度写清楚。
- 所有Query不全量落盘，只保留聚合分数、逐GT统计和最多16张完整预测；不无节制导出Encoder。

## 重跑和恢复

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate {ARTIFACTS}/env
cd {REPO}
bash b0/setup_env.sh                       # 环境准备记录，可重用已有环境
python -m b0.prepare_data                   # 幂等数据构建及全量检查
python -m b0.check_ops                      # CUDA算子检查
python -m b0.smoke                          # 小规模训练和接口一致性
python -m b0.analyze --checkpoint {smoke_checkpoint} --smoke-limit 32
python -m b0.train                          # 完整41 epoch；存在rolling checkpoint时自动恢复
python -m b0.train --resume {ARTIFACTS}/checkpoints/task1/checkpoint.pth
bash b0/eval_task1.sh                       # 官方全量评估入口
python -m b0.analyze --checkpoint {ARTIFACTS}/checkpoints/task1/checkpoint0040.pth
python -m b0.write_reports
bash b0/launch.sh                          # setsid后台串联训练→评估→分析→报告，锁防重复启动
```

训练在单GPU通过官方模型/criterion与相同AdamW参数组执行，外部训练器增加梯度累积、原子Checkpoint、RNG状态、内存及耗时记录；中断优先恢复有效rolling权重。中途恢复的worker增强随机流不保证与不中断执行逐位相同。训练/评估失败由pipeline保留日志和错误状态。

Git分支：`{branch}`；报告生成时Commit：`{commit}`。实际训练Commit记录在Checkpoint/config日志。没有把数据、权重、Conda环境或大日志提交Git。
'''
    progress=f'''## 当前进度

训练状态：`{status.get('state','尚未启动')}`。

```json
{json.dumps(status,ensure_ascii=False,indent=2)}
```

Smoke实际训练图像：{smoke.get('train_images') if smoke else '待运行'}；Optimizer更新：{smoke.get('optimizer_updates') if smoke else '待运行'}；峰值显存：{smoke.get('peak_memory_mib') if smoke else '待运行'} MiB。

正式训练日志：`logs/train.log`；官方评估：`logs/official_eval.log`；分析：`logs/query_analysis.log`；流水线：`logs/pipeline.log`；继续执行 `bash b0/launch.sh`（已有进程时锁会阻止重复）。
'''
    if final:
        metrics=final['official_metrics'];miss=final['unknown_misses']
        rows=[]
        for group in ['known','unknown']:
            rows.append([group+' 全Query']+[f"{final['coverage'][group]['all'][str(t)]*100:.4f}%" for t in [.3,.5,.7]])
            for rank in ['latent','classifier_unknown']:
                for k in [10,50,100]:
                    rows.append([f'{group} {rank} Top{k}']+[f"{final['coverage'][group][rank][str(k)][str(t)]*100:.4f}%" for t in [.3,.5,.7]])
        result=f'''## 完整复现与Query结果

完整分析结果存在：`query_analysis/final/summary.json`；Checkpoint及其SHA256见该文件。

{table(['官方指标','实际值','论文参考'],[['Known mAP@0.5',metrics['K_AP50'],61.4],['U-Recall@0.5',metrics['U_R50'],23.6],['WI (代码返回recall0.8)',metrics['WI'],'需留意论文recall口径'],['A-OSE (代码键AOSA)',metrics['AOSA'],3102]])}

Known mAP差值：{metrics['K_AP50']-61.4:+.4f}个百分点；U-Recall差值：{metrics['U_R50']-23.6:+.4f}个百分点。

唯一图片Unknown非difficult GT：{final['unknown_gt_unique_non_difficult']}；Known：{final['known_gt_unique_non_difficult']}；官方Unknown逐行分母：{final['unknown_gt_official_row_denominator']}。

{table(['候选集','IoU>=0.3','IoU>=0.5','IoU>=0.7'],rows)}

A/B统计（分母需区分）：

```json
{json.dumps(miss,ensure_ascii=False,indent=2)}
```

物体性分布的均值/分位数：summary.json/objectness_distributions；互斥Query组数：{final['query_group_counts']}。图像在 query_analysis/final/figures/ 中。

## 依据实验的后续判断

漏检Unknown中A占{miss['A_fraction_missed_unknown']*100:.2f}%，B占{miss['B_fraction_missed_unknown']*100:.2f}%；以这一口径，{'已有候选但未作为unknown保留的情况占比较高' if miss['A_fraction_missed_unknown']>miss['B_fraction_missed_unknown'] else '原始query缺乏有效定位的情况占比较高'}。

A为评分/类别判别研究提供候选基础，但不能证明重建误差评分有效，也不能把A全部归因于物体性偏低；需要结合已输出的latent与classifier分布、候选Query得分和后续独立实验验证。B部分提示单纯重排已有框存在定位上限。S1/S2后续应先比较空间表征对漏定位B与已定位A的可分性和空间支持，保持已知检测约束；本次未实现这些方法。
'''
    else:
        result='''## 完整复现与Query结果：尚未完成

目前没有可用于完整B0结论的最终41 epoch权重和全测试集Query结果。Known mAP、U-Recall、唯一Unknown GT覆盖率、TopK、A/B比例和物体性分布均不填入伪造数值。Smoke数据仅用于验证流程。

论文存在可比结果：[OWOBJ CVPR2025 Table1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Open-World_Objectness_Modeling_Unifies_Novel_Object_Detection_CVPR_2025_paper.pdf) 的 M-OWODB Task1 PROB+OWOBJ：Known mAP@0.5=61.4，U-Recall=23.6。补充材料Table5：WI=0.03951，A-OSE=3102。这只是参考值，不是本机实测。官方代码 WI 返回 recall0.8，完整日志另有所有recall级别，不在未核对口径时直接比较。

现阶段不能判断定位不足与筛选评分谁为主，也不能断言重建误差物体性有效或存在多大收益。须等待完整权重、官方评估和Query统计。
'''
    (ARTIFACTS/'B0_环境与数据准备报告.md').write_text('# B0 环境与数据准备报告\n\n'+base+'\n'+progress)
    (ARTIFACTS/'B0_实验报告.md').write_text('# B0 PROB+OWOBJ 实验报告\n\n'+progress+'\n'+result+'\n'+base)
    print('Reports updated',now,flush=True)


if __name__=='__main__':
    main()
