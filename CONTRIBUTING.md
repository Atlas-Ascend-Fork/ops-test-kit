# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为"简单的bug修复"，亦可通过提交Issue进行方案讨论。

开发者贡献场景主要包括：

## 一、贡献测试用例

TTK 是昇腾 NPU 算子测试框架，欢迎贡献各模式的测试用例（Kernel/GEIR/ACLNN/E2E）。

### 1. 创建Issue需求

新建`Requirement|需求建议`类Issue，说明新增测试用例覆盖的算子、测试模式及设计思路。

请在提交的Issue中评论`/assign @yourself`认领该任务。

### 2. 需求评审

Sig组将指派Committer对您提交的Issue进行评审并反馈修改意见。请在完成修改后，于Issue中@对应Committer。

### 3. PR提交

测试用例交付件如下：

```text
examples/case_store/
├── kernel/                    # Kernel模式用例
│   └── ${op_name}.csv         # 测试用例CSV
├── geir/                      # GEIR模式用例（字段同Kernel，另支持dyn_input_shapes）
│   └── ${op_name}.csv|.xlsx
├── aclnn/                     # ACLNN模式用例
│   └── aclnn_${op_name}.csv
└── e2e/                       # E2E模式用例
    └── torch_${op_name}.csv   # tf_${op_name}.csv 为 TensorFlow 用例
```

若算子需要自定义Golden或输入生成逻辑，需提供插件文件，与本仓库示例放置方式一致（如 `examples/case_store/kernel/golden.py`）：

```text
examples/case_store/${mode}/
└── golden.py                  # TestSpec（golden/customize_inputs/tolerance 等）
```

> 外部算子仓库贡献用例时，插件文件按该仓库约定放于 `tests/assets/`（`golden.py` / `input.py`），详见 `.claude/skills/ttk-how-write-plugin/SKILL.md`。

PR上库要求：

- 用例文件：CSV格式符合TTK规范，字段名、数据类型、shape等填写正确，可通过对应模式`python3 -m ttk <mode> -i cases.csv --validate`验证（`ttk list` 仅列出用例名，不做字段校验）。
- 插件文件（如有）：遵循TestSpec规范，参考`.claude/skills/ttk-how-write-plugin/SKILL.md`。
- 精度要求：用例需通过精度校验，Golden实现正确，容差设置合理。
- PR提交：通过`git`命令提交目标分支PR，检查PR标题是否清晰、PR描述是否规范（指明更改内容和原因、是否关联对应Issue）、是否签署CLA。

### 4. CI门禁

通过评论`compile`指令触发开源仓门禁，并依据CI检测结果进行修改。

门禁通过后，请在关联的Issue中@指派的Committer。

### 5. Committer检视

Committer检视后将反馈检视意见，请根据意见修改，完成后@指派的Committer。

### 6. Maintainer合入

Committer检视通过后，标注`/lgtm`标签。Maintainer将在1天内进行最终审核，确认无问题后，将标注`/approve`标签合入PR。

## 二、Bug修复

如果您在本项目中发现了Bug，欢迎您新建Issue进行反馈和跟踪处理。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Bug-Report|缺陷反馈`类Issue对Bug进行描述，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行处理。

## 三、功能优化

如果您对本项目某些功能有增强或优化思路，欢迎提出。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Requirement|需求建议`类Issue对优化点进行说明，并提供您的设计方案，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您进行跟踪优化。

## 四、文档纠错

如果您在本项目中发现某些文档描述错误，欢迎您新建Issue进行反馈和修复。

您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建`Documentation|文档反馈`类Issue指出对应文档的问题，然后在评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您纠正对应文档描述。

## 五、帮助解决他人Issue

如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

如果对应Issue需要进行代码修改，您可以在Issue评论框中输入"/assign"或"/assign @yourself"，将该Issue分配给您，跟踪协助解决问题。
