---
project: Ferry
created: 2026-06-10
updated: 2026-06-12 (DEC-014 Korean-weighted Procrustes 반영)
---

# TODO

> TODO↔Issue 연동 정책: **off** (git 저장소 아님, 단일 파일 PoC). 이슈 연동은 제안만.

## High Priority

- 없음 — toy core + Aster 확장(전이/KD/embed/정렬) 핵심 구현·검증 완료.

## Medium Priority — Aster 확장 후속 (사용자 결정 대기)

- [ ] **on-distribution KD/학습** — Aster 한글 embed의 *meaning alignment*(유창성)에 필요한 유일한 진짜 해법. **현 data-free 제약 위반 → 사용자 별도 승인 필수** / DEC-011·DEC-013 근거 (random-probe KD top1=0 천장, closed-form 정렬은 top-k reachability까지만)
- [ ] **byte-order 보존 embed seeding** — 현 byte-composition은 byte MEAN(순서 손실, crude init). 순서 보존 합성(예: 위치 가중)으로 한글 변별력 개선 검토 / data-free 유지 가능 / DEC-012 한계
- [~] **앵커 가중 Procrustes** — 부분 달성(DEC-014, `--kr-weight 50`): median best_kr_rank 9→2, 첫 greedy-Korean 1개. **단 2/10 회귀는 미해소**(`옛날 옛적에`·`대한민국의 수도는` 잔존) → 회귀 완화는 학습/KD 영역으로 남음

## Medium Priority — toy core

- [ ] `ferry_advance.py` — 더 작은 student용 activation-aware 저차원 전이(teacher 활성 PCA 부분공간 투영). DEFERRED, 프로토타입 검증됨(raw-init/tanh에서 이점, align_hidden 이후엔 무의미) / DEC 필요
- [ ] tokenizer 문자열 기반 `t_for_s` 빌더 — 현재 stage-0 기본은 shared-prefix(데모용), 배포는 실제 매칭 맵 필요
- [ ] permutation alignment 도입 검토 — 이름 무관 weight 매칭(Git Re-Basin류)으로 Stage 1 확장 / DEC 필요
- [ ] 다층/트랜스포머 전층(attention·LayerNorm) closed-form 정렬 — 현재는 flat MLP 은닉층 + 마지막 Linear만 / 자기회귀 잔차 근본 축소
- [ ] Qwen plateau 완화 실험 — 더 긴 CPU distill / target·probe 방식 분석 (CPU·데이터-free·GPU 금지 유지)

## Low Priority

- [ ] non-MLP 아키텍처(예: attention) toy 데모 추가
- [ ] capacity sweep 결과를 JSON으로 export하는 옵션

## Completed — Aster 확장 아크 (../SLM_FROM_BEGIN Rust SLM 대상, 신규 파일만 출력, live 학습 미접촉)

- [x] (2026-06-12) **Korean-weighted Procrustes** (`align_aster_embed.py --kr-weight 50`) — 한글 앵커 1826개를 50× 업웨이트해 직교 회전을 한글 친화로. median best_kr_rank 9→2, mean kr_mass 0.646→0.721, 첫 greedy-Korean(`참고`), top-k40 엔드투엔드 한글 다수(`투명하게/번호/데이트코스`). `--kr-weight` default 1.0=DEC-013 재현. 한계: 2/10 회귀 잔존·generic·비유창 / DEC-014
- [x] (2026-06-12) **b′ closed-form Procrustes embed-basis 정렬** (`align_aster_embed.py`) — byte-comp embed를 Aster hidden 기하에 직교 회전 정렬. best_kr_rank 95→9, top-k40 한글 생존 1/10→8/10, Rust chat 기본 top-k40에서 한글 등장. `ferry_aster.AsterForCausalLM.final_hidden()` 노출 / DEC-013
- [x] (2026-06-12) **byte-composition embed seeding** (`transfer_gemma_to_aster.py --embed-byte-compose`) — 미매칭 한글 토큰을 Gemma `<0xXX>` byte-fallback MEAN으로 seed. embed 커버리지 19880→47080(한글 1826→27345), next-token 한글 mass 2%→91%, 샘플링 0/30→30/30 / DEC-012
- [x] (2026-06-12) **Aster PyTorch 재현 + Gemma-2B data-free KD** (`ferry_aster.py`) — Rust forward를 PyTorch로 재현(parity 바이트 일치), `ferry.distill` 소규모 KD. cosine -0.08→0.77이나 top1=0(천장), `SparseVocabMap`으로 49GB dense 회피 / DEC-011
- [x] (2026-06-12) **embed vocab-map 전이** (`--embed-vocab-map`) — tokenizer 문자열 매칭(byte-level 정규화 41.4%). embed=0 collapse 타파(2B≠9B 출력 분기) / DEC-010
- [x] (2026-06-12) **양측 직교 SVD 투영 + 9B 실험** (`ferry._svd_project` 교체) — U_mᵀ A V_n(Eckart-Young). FFN energy 2B=87.8% vs 9B=70.2%(big teacher 불리). layer-select uniform/front / DEC-009
- [x] (2026-06-11) **Gemma-2-2B → aster-1b 순수 가중치 전이** (`transfer_gemma_to_aster.py`) — KD 없는 weight-space algebra. coverage 0.9958, FFN energy 87.8%, schema PERFECT MATCH(236). 정직 결론=작동 모델 아닌 초기 skeleton / DEC-008

## Completed — toy core

- [x] (2026-06-11) `ferry_qwen.py` 실모델 확장 — 실제 Qwen3-0.6B → 아키텍처 변경 ferry-0.1B(103M) CPU·데이터-free 증류, gated 테스트 6종 / DEC-007
- [x] (2026-06-11) `theory.html` 공학자 판본 재작성 — §0 표기법 + 수식·스텝별 설명 주입(무의존성), 구조 검증
- [x] (2026-06-11) combined-mismatch 최악 케이스 데모+회귀 테스트 — vocab+depth+width 동시 + scrambled 맵, 테스트 40→42
- [x] (2026-06-11) Stage 0 어휘 정합 `reconcile_vocab`/`VocabMap`/`shared_token_probe` 구현 — 다른 vocab teacher/student 정합 / DEC-006
- [x] (2026-06-11) Stage 3 `distill` gradient distillation 구현 — 비선형/자기회귀 한계 닫기 / DEC-005
- [x] (2026-06-11) Stage 2b `align_hidden` 은닉층 closed-form 정합 — 비선형 teacher 지원
- [x] (2026-06-10) Stage 1 weight transfer 구현 (`extract_spec`/`match_tensors`/`transform_tensor`/`transfer`/`report`)
- [x] (2026-06-10) Stage 2 output alignment 구현 (`synthetic_probe`/`agreement`/`align_output`)
- [x] (2026-06-10) `clone.py`→`ferry.py`, `test_clone.py`→`test_ferry.py` 리네임 (import alias `import ferry as clone` 유지)
- [x] (2026-06-10) 15 pytest cases 통과, demo 정상 실행
- [x] (2026-06-10) `theory.html` 7섹션 작성·검증
- [x] (2026-06-10) `AGENTS.md` 갱신 (same-answer goal, synthetic-probe 허용)
