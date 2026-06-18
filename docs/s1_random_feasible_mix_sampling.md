# S1 random feasible 混合训练采样规则

更新时间：2026-06-09

本文用于同步 Ctrl-World 训练侧：在 5 个 S1 task 上，把 `expert`、`pca perturbed`、`raw perturbed`、`random feasible` 四类数据混入训练时，如何划分 train/test，以及如何设置采样比例。

## 1. 固定 task

```text
click_alarmclock
click_bell
place_object_basket
open_laptop
stack_blocks_two
```

## 2. Train / test split

只做 train / test，不设 val。

```text
train source episodes: 0-39
test source episodes:  40-49
```

split 单位是 `task + source expert episode`，不是 chunk/window。

同一个 source expert episode 下的所有数据必须属于同一个 split：

```text
expert trajectory
pca perturbed replay samples
raw perturbed replay samples
random feasible replay trajectories
all sliding windows from the above
```

训练数据只用 source episodes `0-39`。测试 manifest 冻结使用 source episodes `40-49`，同一套 test manifest 给不同模型做 paired comparison。

当前已生成的 5-task random feasible formal 数据主要覆盖 source episodes `0/1`，因此它可直接作为 train 数据；如果后续要把 random feasible 加入 test eval，需要另外生成 source episodes `40-49` 的 random feasible test 数据。

## 3. 数据 family

训练混合四类 family：

```text
expert
pca_c8_sigma0p05
raw_sigma0p0025
random_feasible_300step
```

random feasible 使用：

```text
data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk/
  rf_5task_300step_2ep5start_formal_uniform_10seed_v1
  rf_5task_300step_2ep5start_formal_weighted_10seed_v1
```

训练时把 uniform / weighted 合并为同一个 `random_feasible_300step` family；family 内部建议 uniform / weighted 以 `1:1` 采样。

## 4. 推荐采样概率

推荐训练 dataloader 采用两级采样：

```text
先采 family -> 再在 family 内采 task / trajectory / window
```

family 概率如下。

### 主实验：3:1:1:1

```yaml
family_sampling:
  expert: 0.500000
  pca_c8_sigma0p05: 0.166667
  raw_sigma0p0025: 0.166667
  random_feasible_300step: 0.166667
```

该设置让 expert 仍占 50%，三类 off-expert 合计 50%。这是当前最推荐的 S1+RF 主实验配置。

### 激进对照：1:1:1:1

```yaml
family_sampling:
  expert: 0.250000
  pca_c8_sigma0p05: 0.250000
  raw_sigma0p0025: 0.250000
  random_feasible_300step: 0.250000
```

该设置更强调 off-expert，对 expert 行为保持可能更有风险，建议作为 ablation。

## 5. Family 内采样

建议 family 内部继续做均衡：

```text
task uniform
variant uniform/weighted 1:1 for random feasible
trajectory/sample uniform
window uniform within selected trajectory/sample
```

不要把所有 sliding windows 直接摊平成一个大池后全局 shuffle。random feasible 是 300-step 长轨迹，直接摊平会产生大量相邻强相关窗口，并让 RF 在自然样本数上压过其他 family。

## 6. 静态 manifest 数量参考

如果训练代码暂时不能做 family-balanced sampler，只能提前生成静态 manifest，可以按下面数量抽样。

### chunk size = 16

| target | expert | pca | raw | random feasible |
|---|---:|---:|---:|---:|
| 3:1:1:1 | 34,214 | 11,405 | 11,405 | 11,405 |
| 1:1:1:1 | 34,214 | 34,214 | 34,214 | 34,214 |

### chunk size = 32

| target | expert | pca | raw | random feasible |
|---|---:|---:|---:|---:|
| 3:1:1:1 | 31,014 | 10,338 | 10,338 | 10,338 |
| 1:1:1:1 | 31,014 | 31,014 | 31,014 | 31,014 |

chunk size 32 的 `1:1:1:1` 会需要重复采样 pca/raw，因为有效 pca/raw 池小于 31,014。若不允许重复采样，则 chunk size 32 的 `1:1:1:1` 最多只能取四类各 13,360 条。

## 7. 当前有效窗口池口径

以上数量基于 5 task train split 的有效数据估算：

| family | chunk=16 windows | chunk=32 windows |
|---|---:|---:|
| expert | 34,214 | 31,014 |
| pca c8 | 364,616 | 21,448 |
| raw s0025 | 227,120 | 13,360 |
| random feasible 300 | 254,505 | 240,217 |

换算规则：

```text
pca/raw: 32-step replay sample -> chunk16 有 17 个窗口，chunk32 有 1 个窗口
RF300:   300-step trajectory -> chunk16 有 285 个窗口，chunk32 有 269 个窗口
```

训练报告中必须明确使用的是 `3:1:1:1` 还是 `1:1:1:1`，以及采样是 family-balanced 还是静态 manifest 抽样。
