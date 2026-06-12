---
project: Ferry
created: 2026-06-10
updated: 2026-06-12
current_phase: 6
---

# Plan

## Current Phase

Phase 6: 실모델 확장 + 공학자 문서 (완료). 핵심 파이프라인(Stage 0–3), 실모델(`ferry_qwen.py`),
`theory.html` 공학자 판본까지 완료·검증. Phase 7(정렬 일반화)은 선택·미착수.

## Phases

### Phase 1: Weight-space 전이 (완료)

- [x] `extract_spec` / `match_tensors` (이름 기반 페어링)
- [x] `transform_tensor` 4-kind 분기 (Copy / CropPad / SvdProject / Skip)
- [x] `transfer` + `report` (coverage, mean_relative_error, by_kind)
- [x] Stage-1 테스트 10종

### Phase 2: Closed-form 출력 정렬 (완료)

- [x] `synthetic_probe` (랜덤 텐서, 데이터셋 없음)
- [x] `agreement` (mse / top1_agree / cosine)
- [x] `align_output` (last Linear hook → `lstsq` → weight/bias write-back)
- [x] capacity sweep로 rank condition 실증
- [x] Stage-2 테스트 5종

### Phase 3: 문서화 (완료)

- [x] `AGENTS.md` — same-answer PoC 설명, hard constraints, gotchas
- [x] `theory.html` — self-contained 이론 문서
- [x] `.agents/` 6 docs 초기화 + `handoff.md`

### Phase 4: 비선형·자기회귀 한계 닫기 (완료)

- [x] Stage 2b `align_hidden` — 은닉층 closed-form 정합(비선형 teacher 지원)
- [x] Stage 3 `distill` — gradient distillation(DEC-005), 데이터-free, 매 step 새 probe
- [x] `ActMLP`/`TinyLM` 데모 + distill 비교, 테스트 ~32

### Phase 5: 어휘·세 축 불일치 (완료)

- [x] Stage 0 `reconcile_vocab`/`VocabMap`(DEC-006) — 다른 vocab 정합
- [x] combined-mismatch 최악 케이스 데모+회귀 테스트(vocab+depth+width+scrambled 맵), 테스트 40→42

### Phase 6: 실모델 + 공학자 문서 (완료)

- [x] `ferry_qwen.py` — 실제 Qwen3-0.6B → 아키텍처 변경 ferry-?B, CPU·데이터-free(DEC-007)
- [x] `test_ferry_qwen.py` gated 테스트 6종
- [x] `theory.html` 공학자 판본 — §0 표기법 + 수식·스텝별 설명(무의존성)

### Phase 7 (선택, 미착수): 정렬 일반화 + 효율 전이

- [ ] `ferry_advance.py` — 더 작은 student용 activation-aware 저차원 전이(DEFERRED, 프로토타입 검증됨)
- [ ] tokenizer 문자열 기반 `t_for_s` 빌더(stage-0 배포용)
- [ ] permutation alignment(이름 무관 매칭) / 트랜스포머 전층·attention 정합
- [ ] Qwen plateau 완화 실험 / capacity sweep JSON export / non-MLP toy 데모

## Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Stage 1 weight transfer | 2026-06-10 | done |
| Stage 2 output alignment | 2026-06-10 | done |
| Theory doc + project docs | 2026-06-10 | done |
| Stage 2b/3 (align_hidden, distill) | 2026-06-11 | done |
| Stage 0 vocab + combined mismatch | 2026-06-11 | done |
| ferry_qwen 실모델 + theory 공학자 판본 | 2026-06-12 | done |
| Phase 7 정렬 일반화 | — | 선택, 미착수 |
