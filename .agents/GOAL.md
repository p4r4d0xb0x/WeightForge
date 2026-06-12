---
project: Ferry
created: 2026-06-10
updated: 2026-06-12
---

# Goals

## Vision

서로 다른 layer 수 / hidden dimension / vocabulary를 가진 **Teacher**와 **Student** 모델이 *동일한 입력에 대해 동일한 답*을 내도록, **학습 데이터 없이**(합성 probe만) 어휘 정합 + weight-space 전이 + closed-form 정렬 + gradient distillation으로 맞추는 PoC. 실제 Qwen3-0.6B에서도 아키텍처 변경 전이를 시연.

## Objectives

- [x] Stage 0: 어휘 정합 `reconcile_vocab`/`VocabMap` — 다른 vocab teacher/student 정합 (DEC-006)
- [x] Stage 1: 이름 기반 weight 전이 (Copy / CropPad / SvdProject / Skip) — 결정론적 algebra
- [x] Stage 2: synthetic probe 기반 closed-form 출력 정렬 (`torch.linalg.lstsq`)
- [x] Stage 2b: `align_hidden` 은닉층 closed-form 정합 — 비선형 teacher 지원
- [x] Stage 3: `distill` gradient distillation — 비선형/자기회귀 한계 닫기 (DEC-005, 데이터-free)
- [x] same-answer 보장의 **조건(rank condition)** 을 정직하게 측정·보고
- [x] 실모델 확장 `ferry_qwen.py` — 실제 Qwen3-0.6B → 아키텍처 변경 ferry-?B, CPU·데이터-free (DEC-007)
- [x] `ferry.py` + `test_ferry.py`(42) + `theory.html`(공학자 판본) + 실모델 파일·테스트

## Success Criteria

- 선형/충분 width: held-out probe `top1_agree == 1.0`, `mse ≈ 0` (실측 ~8e-15).
- 비선형 depth-matched: `align_hidden` ~0.97 → `distill` ~0.99.
- Bottleneck width / 자기회귀 잔차 등 한계를 **숨기지 않고** residual로 보고(가짜 일치 없음).
- `python -m pytest test_ferry.py -q` **42 cases** pass, `python ferry.py` demo(6-part) 정상.
- `python -m pytest test_ferry_qwen.py -q` 6 cases(gated) pass, `python ferry_qwen.py` 실모델 리포트 정상.

## Non-Goals

- 학습 데이터 / 데이터셋 / 데이터 로더 / 데이터용 disk I/O (hard constraint, 금지). 합성 probe는 허용.
- GPU 사용 (DEC-007, `ferry_qwen.py`는 CPU·float32 전용).
- ~~gradient training loop 금지~~ — **DEC-005로 해제**. 단 gradient는 stage 3 `distill`에만, stage 1–2는 결정적 closed-form 유지.
- permutation alignment (Git Re-Basin류) / 트랜스포머 전층 정합 — 미래 작업(향후 과제).
- packaging / 멀티 모듈 구조 (단순성이 명시 요구사항; `ferry_qwen.py`만 argparse CLI).
