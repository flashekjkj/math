# 自动项目审计

仅使用现有文件；没有重新推理或训练。

## 数据检查

```json
{
  "splits": {
    "train": {
      "count": 1000,
      "duplicate_ids": 0,
      "duplicate_families": 0
    },
    "valid": {
      "count": 100,
      "duplicate_ids": 0,
      "duplicate_families": 0
    },
    "test": {
      "count": 200,
      "duplicate_ids": 0,
      "duplicate_families": 0
    }
  },
  "family_overlap": {
    "train__valid": 0,
    "train__test": 0,
    "valid__test": 0
  },
  "passed": true,
  "limitation": "Family checks are heuristic; semantic duplicates and question correctness need human review."
}
```

## base-test → sft-test

```json
{
  "neither_confirmed_correct": 143,
  "response_changed": 200,
  "both_confirmed_correct": 14,
  "gained_confirmed_correct": 34,
  "lost_confirmed_correct": 9,
  "n": 200,
  "accuracy_delta_pp": 12.5
}
```

## sft-valid → merged-sft-valid

```json
{
  "neither_confirmed_correct": 82,
  "both_confirmed_correct": 17,
  "response_changed": 8,
  "gained_confirmed_correct": 1,
  "n": 100,
  "accuracy_delta_pp": 1.0
}
```

## merged-sft-valid → dpo-demo-valid

```json
{
  "neither_confirmed_correct": 82,
  "both_confirmed_correct": 18,
  "response_changed": 6,
  "n": 100,
  "accuracy_delta_pp": 0.0
}
```

## 解释边界

自动数值正确不代表推导正确。数据分组不证明无语义泄漏。
审阅队列只用于验证集诊断，不可直接转成训练偏好对。
比较 SFT 合并前后的差异时，需用同一提示、精度和生成参数重新确认。
