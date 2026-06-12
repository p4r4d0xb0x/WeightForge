---
project: Ferry
created: 2026-06-10
updated: 2026-06-12 (DEC-015 공개 git repo 발행 + source-available publication-reserved LICENSE)
decision_count: 15
decisions: [DEC-001, DEC-002, DEC-003, DEC-004, DEC-005, DEC-006, DEC-007, DEC-008, DEC-009, DEC-010, DEC-011, DEC-012, DEC-013, DEC-014, DEC-015]
---

# Decisions

## DEC-015 — 공개 git repo 발행(WeightForge) + source-available, publication-reserved 커스텀 LICENSE

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] 사용자가 빈 공개 GitHub repo `p4r4d0xb0x/WeightForge`(PUBLIC, isEmpty)를 연결하고
  verbatim **"자 이제 해당 프로젝트에 레포를 연결시켜줬음 잘 세팅해봐"** = 초기 세팅 위임.
- [Fact] 그 전까지: git 없음·README 없음·LICENSE 없음(flat PoC). 로컬 `main` 커밋 0개.
- [Fact, 위험] `test_output/` = **23GB**(3.8GB `params.safetensors` × 6) → GitHub 100MB 한도
  초과, 절대 커밋 금지. `.gitignore` 없었음.
- [Fact] 라이선스 의도(사용자 verbatim): **"사용은 허가하지만, 논문 / 연구 실적 등 발표는 나만 가능한"**
  = use 허용 + academic publication·research-credit·공개발표는 Author 독점.
- [Fact] 표준(OSI/SPDX) 라이선스 중 그 의도를 만족하는 것 **없음**: MIT/Apache/BSD/GPL은 발표를
  막지 않고, CC BY-NC·PolyForm Noncommercial은 상업만 제한(발표권 유보 조항 없음).
- [Fact] GitHub 설명문이 실제와 모순: "directly transforming ... **without training or output
  matching**" 인데 실제로는 Stage 2 output matching + Stage 3 gradient training(DEC-005) 사용.

### Decision (결정사항)

1. **`.gitignore` 추가**: `test_output/`, 모델 가중치(`*.safetensors`/`*.bin`/`*.pt`/`*.pth`/
   `*.ckpt`/`*.gguf`/...), 파이썬 캐시(`__pycache__`/`.pytest_cache`/...), venv, 에디터/OS junk
   제외. 커밋 대상 = 소스 5 + 테스트 2 + `theory.html` + `AGENTS.md` + `.agents/` 6 + `README.md`
   + `LICENSE` ≈ **0.3MB**. data-free 불변(가중치 git 미진입 보장).
2. **최소 `README.md` 추가**: 이전 "no README" 규약을 사용자 결정으로 override(단순성·no-packaging
   규칙은 유지). 프로젝트 요약·4-stage·layout·실행법·정직한 한계·LICENSE 안내만.
3. **커스텀 source-available, publication-reserved `LICENSE` (Strict Edition)**: 사용자 verbatim
   **"조항을 더 엄격하게"** 반영. §2 좁은·개인·revocable use(private·비공개·비상업 evaluation/research
   한정); §3 금지 9종을 Author 독점(사전 서면 signed 동의 필요) — publication(학술+비공식)·research-credit·
   redistribution/public fork·commercial·**Work/Output로 다른 모델 train/distill/평가 금지(e)**·dataset
   편입 금지(f)·patent 출원 금지(g)·trademark/branding 금지(h)·competing method 구축 금지(i); §6 아이디어는
   저작권으로 못 막음을 정직히 명시(honest limit)하되 expression 복제는 금지; §7 no implied rights; §8 위반 시
   자동 종료 + 사본/Output 파기 의무; §9 survival; §10 irreparable harm·injunctive relief; §11 무보증.
   **OSI 비승인(의도적)**.
4. **GitHub 설명문 정정**: 실제 파이프라인(data-free weight transfer + closed-form align + gradient
   distill, depth/width/vocab 변경) 반영하도록 `gh repo edit`.
5. **초기 커밋 + origin/main 푸시**. `AGENTS.md` layout/operating-rules의 stale 문구("no git/README",
   "Not a git repo") 동기화.

### Consequences (영향)

- repo가 공개 + 법적으로 표기됨. 23GB 가중치·합성 외 데이터는 git에 진입 불가(data-free 유지).
- "no README" 규약 폐기 → `AGENTS.md` Layout 갱신. DEC-004의 "Not a git repo" 근거 무효화(정책 off는
  유지, 근거 문구만 갱신).
- 라이선스는 코드/텍스트/파생물의 복제·배포·발표는 규율하나 **독립 아이디어는 못 막음**(저작권 한계,
  LICENSE §5에 명시) — 과대 주장 회피.
- copyright holder = **Kim Dogyun (김도균)**(LICENSE 저작권 표기). repo URL의 `p4r4d0xb0x`는 GitHub 핸들로 유지.

### Alternatives (대안)

- **MIT/Apache-2.0**: 기각 — 발표/연구실적 유보 불가(요구 미충족).
- **CC BY-NC / PolyForm Noncommercial / Prosperity**: 기각 — 상업만 제한, 발표권 유보 조항 없음. CC는
  소프트웨어에 부적합.
- **LICENSE 없음(all rights reserved)**: 기각 — "사용 허가"조차 안 됨(사용자는 use 허용 원함).
- **README 없음(규약 유지)**: 기각 — 사용자가 공개 repo용 최소 README 선택.

## DEC-014 — Korean-weighted orthogonal Procrustes (`--kr-weight`): 한글 anchor 업웨이트로 정렬을 한글 친화로, 정직한 추가 개선

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] 사용자 m0453 verbatim: **"조금 더 힘써보자"** = DEC-013 b′ 정렬의 한계(등장 한글 generic,
  2/10 프롬프트 회귀, greedy top1 여전히 비한글)를 data-free로 한 걸음 더 밀어보기.
- [Fact, 가설] DEC-013 Procrustes `R`은 앵커 19880개 중 **한글 1826(9%)에 지배되지 않고** 비한글
  91% 기하에 정렬됨 → 회전이 한글 친화적이지 않아 한글이 generic·diffuse. **한글 앵커를 업웨이트**하면
  소수 한글 방향이 global rotation을 더 끌어 한글 순위가 개선될 것.
- [Fact] 옵션 랭킹(anti-evasion, 보고서에 그대로 명시): **1순위 = on-distribution 실데이터 학습/KD**
  (진짜 천장 돌파이나 **data-free hard constraint 위반 → 사용자 승인 필요, 기본 선택 금지**), 2순위 =
  teacher-generated probe KD(data-free이나 한글은 byte-fallback 41% 벽 + CPU 비쌈), **3순위(이번 실행)
  = Korean-weighted Procrustes**(closed-form, 즉시, 직교성 유지). FFN hidden-align(C)은 teacher/student
  이질성(head_dim 256 vs 96)으로 닫힘형 불안정 → 보류.
- [Constraint] data-free, GPU 금지(DEC-007), live aster-1b 미접촉(신규 파일만).

### Decision (결정사항)

1. **가중 orthogonal Procrustes**: `minimize Σ wᵢ‖xᵢR − yᵢ‖²` → `M_cross = Xᵀ diag(w) Y =
   (X·w[:,None])ᵀ @ Y`, SVD → `R = UVᵀ`. `e_fit_whole`(앵커별 독립 ridge 회귀 target)은 가중치
   무관 → **Procrustes 단계에서만 가중**(surgical change, lstsq target 불변).
2. `align_aster_embed.py` 6군 편집: `procrustes_rotation(x, y, weights=None)`(None=DEC-013 uniform
   완전 하위호환), `align_once(..., anchor_weights=None)`, `run_align`에서
   `anchor_weights = where(kr_mask[whole_ids], kr_weight, 1.0)`, **CLI `--kr-weight`(default 1.0 =
   DEC-013 재현)**, report json에 `kr_weight` 기록. `--iters` 기본 1 유지(과회전 붕괴 경고 존속).
3. **sweep로 sweet spot 확정**: `kr_weight ∈ {1,5,20,50} × iters {1,2}` → **kr_weight=50, iters=1**
   채택. 아티팩트 `aster-1b-from-gemma-2-2b-embedmap-bc-aligned-krw50/`(embed-only delta).

### Consequences (영향)

- [측정, 정직] **DEC-013 대비 실제 개선**(10-prompt harness, survive@40 / median best_kr_rank /
  greedy-top1-Korean / mean kr_mass):
  - bc(DEC-012): 1/10 · 96 · 0 · 0.734
  - bc-aligned(DEC-013, kr_w1): 8/10 · 9 · 0 · 0.646
  - **bc-aligned-krw50(DEC-014): 8/10 · 2 · 1 · 0.721** ← median rank 9→2, 첫 greedy-Korean 1개(`참고`).
- [측정, 정직] 엔드투엔드 top-k40(Rust 기본 디코더 등가, temp0.9): DEC-013 aligned은 거의 영어/기호
  (`Hamp islation XNUMX`) vs krw50은 실제 한글 다수(`투명하게/번호/데이트코스/하계/구성해/보증금`).
- [측정, 정직] **한계(잔여)**: 2/10 프롬프트 회귀 존속(`옛날 옛적에` rank282, `대한민국의 수도는`
  4→580), greedy 대부분 `<pad>`(한글 rank2 바로 위), 등장 한글 generic·반복(`프랑 딱히 elfare` 반복)·
  비유창. **byte-order 손실 + 문법 신호 부재는 정렬로 불가** = 학습/KD 영역(진짜 천장 = on-distribution,
  승인 필요).
- [불변] 직교 R이라 row-norm 보존(byte-comp 균등화 유지), embed-only delta(235 텐서 byte-identical),
  **schema PERFECT MATCH(236, 0/0/0 vs live aster-1b)**, FFN/attn 미변. py_compile OK, test_ferry
  42/42(ferry core 미변), 잔여 마커 0.
- [과회전 재확인] iters≥2는 전반 붕괴(kr_w1 iters2 mass0.029); 가중이 일부 안정화하나 iters1보다 열등
  → DEC-013의 `--iters 1` 고정 결정 유효.

### Alternatives (대안)

- **on-distribution 학습/KD** (가장 완전한 해, 1순위): data-free 위반 → 미실행, 사용자 승인 대기. 거부
  아닌 **보류**(보고서 1순위 명시).
- **teacher-generated probe KD** (2순위, data-free): 한글 byte-fallback 41% 벽 + CPU 비용 → 이번 보류.
- **FFN hidden-basis align (C)**: head_dim 256 vs 96 이질성으로 닫힘형 불안정 → 보류.
- **kr_weight=1.0 (DEC-013 유지)**: 개선 여지 미사용 → 거부. **default는 1.0 유지**(하위호환·재현성).
- **iters≥2 / scaled Procrustes**: 과회전 붕괴 / scale 아티팩트 → 거부(DEC-013과 동일).

## DEC-013 — closed-form orthogonal Procrustes embed-basis alignment (b′): 한글 mass de-diffuse, 정직한 partial positive

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] DEC-012 byte-composition은 한글 **reachability**(embed row 비영, next-token mass 2%→91%)는
  확보했으나 **정렬(alignment)** 은 못 함: seed가 Gemma embed singular basis에 살아 Aster hidden
  기하와 미정렬 → `logit=h·embed_row`가 노이즈 → 91% mass가 27345 한글 토큰에 **diffuse**(최상위
  한글이 rank~95). greedy/top-k40에서 구조 토큰(공백/구두점/영어)이 항상 이김 → Rust에서 한글 보려면
  `--top-k 0` 필요.
- [Fact] 사용자 m0382 verbatim: **"b' 먼저"** = 옵션 (b′) 폐쇄형 부분 정렬(data-free, gradient 없이
  closed-form)을 먼저 구현. 목표: seed basis를 Aster hidden 기하에 회전 정렬해 diffuse를 줄이고 한글
  순위를 개선(가능하면 top-k에서 생존).
- [Fact] tied-embed: `v2.embed.weight`가 입력 임베딩 + 출력 투영 이중역 → 정렬이 양쪽에 영향.
- [Constraint] data-free(synthetic shared-probe + tokenizer vocab만), GPU 금지(DEC-007), live
  aster-1b 미접촉(신규 파일만).

### Decision (결정사항)

1. **신규 파일 `align_aster_embed.py`** (toy core ferry.py 미오염, ferry_qwen/ferry_aster와 같은
   per-concern 분리). Ferry Stage-2b 정신의 폐쇄형 회전:
   - whole-token 매칭 19880개(한글 1826)를 **앵커**로. student를 synthetic **shared-token probe**로
     forward → tied-head 입력 feature `F=final_hidden` 수집. 같은 probe를 remap해 real Gemma-2B
     teacher forward → 앵커 토큰의 teacher logit 열 gather. 정규방정식 `A=FᵀF`(d×d), `B=FᵀT_whole`
     누적(float64, ridge `1e-3·mean(diagA)`) → `E_fit_whole = solve(A,B)ᵀ` = 앵커가 가져야 할
     embed row(Aster 출력 basis).
   - **orthogonal Procrustes**: `M=E_seed_anchorᵀ·E_fit_anchor=UΣVᵀ` → `R=UVᵀ`. `E_aligned=E_seed@R`
     를 **전 48000 row에 동일 적용**(byte-composed 한글 row도 같은 `proj` 행렬에서 나와 같은 basis →
     R이 일관 적용). **직교 R이라 row-norm 정확 보존**(byte-comp norm 균등화=한글 경쟁력 유지).
2. **`ferry_aster.py`에 `AsterForCausalLM.final_hidden()` 추가**(forward가 호출 → parity 보존).
   alignment가 tied-embed에 의존 없이 출력투영을 회귀하도록 feature 노출.
3. **출력은 embed-only delta**: 입력 params(bc) 복사 후 `v2.embed.weight`만 교체 저장 → 나머지 235
   텐서 byte-identical. 출력 `aster-1b-from-gemma-2-2b-embedmap-bc-aligned/`.
4. **`--iters` 기본 1 고정**(과반복 금지, 아래 측정 근거).

### Consequences (영향)

- [측정, 정직] **목표 달성(partial positive)**: 10개 프롬프트 best-Korean-within-top40(=greedy/top-k
  생존) **BEFORE 1/10 → AFTER 8/10**. 대표 프롬프트 best_kr_rank **95→9**(diffuse 급감), kr_mass
  0.91→0.81 유지. Rust slm-cli chat **기본 top-k 40**에서 한글 등장(`스톡/웹툰/국민/KBS/삼성/놨`) —
  bc는 동일 설정에서 한글 0개. 즉 정렬 전 `--top-k 0` 필수였던 한글이 정렬 후 기본 top-k40에서 생존.
- [측정, 정직] **비균일·한계**: 2/10 프롬프트 회귀(`옛날 옛적에` mass 0.916→0.002, `대한민국의 수도는`
  rank 4→732). 7개 프롬프트가 거의 동일한 rank9/mass0.807로 수렴 = 등장 한글이 **generic·prompt-
  비민감**(undertrained crude init). greedy top1은 여전히 비한글. anchor teacher-agreement cosine은
  미미(-0.004→-0.02, top1 0.004→0.008): 직교 R은 scale 못 맞춰 resid>1(scale 아티팩트, ranking 무관);
  실제 정렬 지표 anchor_row_cosine≈0.34(임의 0 대비 양수이나 1과 거리 멂).
- [측정, 정직] **과반복 붕괴**: iters 1→3에서 한글 best_rank 9→94→762, mass 0.81→0.65→0.00. 원인:
  앵커의 91%가 비한글 + teacher가 random probe에서 한글 비선호(영어/구조 prior) → 반복할수록 anti-Korean
  prior로 과회전. ∴ iters=1이 sweet spot(기본값 고정).
- [근본 한계] **byte-order 손실 + 문법 신호 부재는 정렬로 못 고침**(학습/KD 영역). data-free 폐쇄형
  정렬로 닫을 수 있는 것은 hidden-basis 미정렬(failure point 1)뿐 — 한글이 **top-k에 도달**하나
  **유창하지 않음**. 즉 b′는 "reachability under top-k"를 달성, "alignment of meaning"은 미달.
- 검증: schema PERFECT MATCH(236, 0/0/0 vs live), embed-only delta 확인, py_compile OK, test_ferry
  42/42(parity refactor 무해). 잔여 마커 0.

### Alternatives (대안)

- **거부: align_output/align_hidden 직접 재사용** — `_last_linear`가 tied-embed에서 ffn_down을 잡음
  (embed는 nn.Parameter, LM head는 `x@embed.t()`). 전용 closed-form 필요.
- **거부: scaled Procrustes(R=s·orthogonal)** — 균일 scale은 softmax 온도만 바꾸고 ranking 불변 +
  byte-comp norm 균등화 파괴. 직교 R 채택.
- **거부: 한글 앵커만으로 R 학습 / 한글-aware target** — data-free 한글 whole-token teacher target이
  구조적으로 부재(Gemma는 한글을 per-position byte-fallback로만 출력, 단일 위치 whole-한글 logit 없음).
  teacher-anchored가 유일한 원칙적 closed-form target이고 그것은 anti-Korean. 비한글 다수 앵커로 학습한
  R을 한글에 전이하는 현 방식이 최선.
- **보류: on-distribution KD/학습(진짜 의미 정렬)** — data-free 제약 위반, 사용자 별도 승인 필요(DEC-011
  random-probe KD는 top1=0 천장 확인). b′는 그 전 단계의 closed-form 천장을 측정·달성.

## DEC-012 — byte-composition embed seeding으로 한글 커버리지 강화 (whole-token 매칭의 byte-fallback 보강, 정직한 partial positive)

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] DEC-010 embed vocab-map은 whole-token 문자열 일치만 사용 → 한글 row를 1826개만 seed.
  사용자가 next-action 옵션 중 **(b) "tokenizer 한글 토큰 매칭 강화로 embed 커버리지 상승"**을 선택
  ("b 먼저 시도").
- [Fact, 진단] 근본 원인은 정규화 버그가 아님. Aster(student) tokenizer는 GPT-2 byte-level BPE로
  whole 한글 토큰 27345개 보유하나, Gemma-2-2b(teacher) tokenizer는 SentencePiece로 한글을 대부분
  **byte-fallback(`<0xXX>`, 255개, 0x09 TAB만 없음)**으로 토큰화 → whole 한글 토큰이 ~2295개뿐.
  NFC/NFD 무용, NFKC는 +1 한글(무의미). Aster 한글 토큰 25519개가 어떤 정규화로도 whole-match 불가.

### Decision (결정사항)

1. **`transfer_gemma_to_aster.py`에 byte-composition seeding 추가**(embed seeding 전용, opt-in
   `--embed-byte-compose`, `--embed-vocab-map` 필수). 미매칭 student 토큰의 UTF-8 바이트 각각을
   Gemma `<0xXX>` byte-fallback 토큰으로 매핑 → 그 byte 임베딩들(hidden축 right-projected)의
   **MEAN**으로 embed row seed. 모든 byte가 fallback에 있을 때만 컴포즈(부분 seed 거부 = 정직).
2. **norm rescale**: tied embed에서 logit = h·row이므로, byte-composed row를 whole-token row의
   평균 norm으로 재스케일 → 한글 row가 greedy에서 경쟁 가능. (측정: nonzero row-norm mean 1.52 std 0.09.)
3. **scope 격리**: `build_vocab_map`(KD logits SparseVocabMap의 SSOT)은 **미변경** —
   byte-composition은 logit 차원에 무관(whole-token 단위)하고 DEC-010/011 재현성을 깨면 안 됨.
   byte-comp은 그 위에 additive로만 추가. `ferry_aster.py` import 시그니처 불변.

### Consequences (영향)

- [정직, positive] **한글 reachability 결정적 상승**: embed 커버리지 19880→**47080/48000**(한글
  1826→**27345**), next-token 한글 prob mass **2%→91%**(prompt '옛날 옛적에 한 마을에'),
  full-distribution 샘플링(temp 1.0 ×30) 한글 **0/30→30/30**(실제 단어: 협정/찌개/현장을/요인…),
  Rust 런타임 slm-cli도 `--top-k 0`에서 한글 생성 확인('항공편을 예상했다 성장하고…'). schema
  PERFECT MATCH(236), FFN energy 87.81% 불변.
- [정직, 한계] (1) mass가 27345개 한글 토큰에 **diffuse** → greedy/top-k40은 여전히 structural
  (space/구두점)/영어 선택, 한글을 보려면 **`--top-k 0` 필수**(기본 top-k 40은 한글 은폐).
  (2) 한글 토큰이 개별적으로 plausible하나 **문법적 coherent는 아님**(fluency는 training/KD 필요).
  (3) byte-mean은 byte ORDER 손실 → **crude non-zero INIT**(Gemma byte signal은 담지만 whole-token
  semantics는 아님). fluency 주장 아님.
- [결론] 목표(한글 embed 커버리지 상승)는 달성. byte-composition이 "산 것"(한글이 분포에 존재 =
  reachable)과 "못 산 것"(diffuse·비유창)을 모두 실측. embed=0 collapse(DEC-009) → whole-token
  seed(DEC-010) → byte-composition(DEC-012)으로 한글 가시성이 단계적으로 개선됨.

### Alternatives (대안)

- **NFKC 정규화 강화**: 거부 — 진단 결과 +1 한글뿐(근본 원인은 byte-fallback이라 정규화로 해결 불가).
- **`build_vocab_map`에 byte-comp 통합**: 거부 — logit map(KD)을 오염시키고 DEC-010/011 재현성 파괴.
  embed seeding 경로에만 additive로 격리.
- **부분 byte seed 허용**(일부 byte만 fallback에 있어도 컴포즈): 거부 — 불완전 토큰을 왜곡 표현.
  모든 byte가 fallback에 있을 때만 컴포즈.
- **byte order 보존(가중합/위치 인코딩)**: 보류 — data-free·closed-form 범위 밖. crude mean으로 충분히
  reachability 입증, 그 이상은 training 영역.

## DEC-011 — Aster forward의 PyTorch 충실 재현 + 실제 Gemma-2B data-free KD (정직한 negative result, fluency 천장 실측)

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] DEC-008~010으로 순수 가중치 전이는 collapse만 깨고 fluent 출력은 못 냄을 확인.
  사용자가 "그럼 여기에 KD를 하면 좋아질까?"를 물었고, 사전 분석(toy worst-case distill 0.86,
  실모델 Qwen 0.6B random-probe KD 0.00→0.20 천장, Aster는 더 가혹)을 제시.
- [Fact] 사용자가 question에서 **'실제 Aster PyTorch 재현 + Gemma-2B KD (완전한 답)'**을 명시 선택
  — toy proxy나 "분석으로 충분"이 아니라 실제 Aster에 KD 적용해 천장 실측을 요구.
- [Fact] Aster는 Rust 구현이라 autograd 불가 → `ferry.distill`(PyTorch Adam loop)을 쓰려면
  Aster forward를 PyTorch `nn.Module`로 재현해야 함. data-free(synthetic token probe만)·GPU 금지 유지.

### Decision (결정사항)

1. **`ferry_aster.py` 신규 파일**(toy core 오염 금지, AGENTS.md): Aster forward를 PyTorch로 정확 재현
   — dual-RoPE(interleaved GPT-J, local θ1e4/global θ1e6, full head_dim 96), hybrid attn
   (layer i%5==4 + 마지막 layer global, 나머지 sliding window 512), GQA 16/8, attn soft-cap 50,
   GeGLU(gelu-tanh), RMSNorm(**raw gamma, Gemma의 1+γ 아님**), final soft-cap 30, tied embed
   (**no sqrt scaling**, Gemma와 다름). `transfer_gemma_to_aster.py`의 safetensors I/O·vocab map 재사용.
2. **parity를 KD의 전제 조건으로 확증**: PyTorch 재현이 Rust 런타임과 다르면 KD가 무의미하므로,
   `slm-cli generate --top-k 0 --temperature 0.0 --repetition-penalty 1.0`(순수 greedy)와
   바이트 단위 일치를 두 프롬프트로 검증(통과). slm-cli **chat은 top-k=40 기본값 때문에 parity 부적합**.
3. **메모리**: `ferry.build_vocab_map`의 dense (256000,48000) projection은 ~49GB라 실모델 불가 →
   `SparseVocabMap`(index_select gather)로 회피.
4. KD는 **Gemma-2B teacher만**(9B는 CPU 비현실), 소규모(batch 8, seq 16, steps 100).

### Consequences (영향)

- [정직] **logit-space는 개선, 그러나 fluency 천장은 못 넘음**: held-out probe agreement
  before top1=0.0/mse=109.95/cosine=-0.08 → after top1=0.0/mse=10.72/cosine=0.77.
  cosine은 크게 올랐으나 **top1은 probe 위에서조차 0** 유지.
- [정직] **chat 출력은 더 유창해지지 않음(오히려 더 degenerate)**: before(전이) =
  인식가능 영단어 다수(call/march/phone/schedule…) vs after(KD) = 파편·숫자·구두점.
  근본 원인: KD가 **random token probe**(off-distribution)에서만 teacher를 모방 → 실프롬프트
  greedy 디코딩으로 전이 안 됨. 용량 10%+vocab 41%+tied-embed basis 미정렬이 겹침.
- [결론] data-free random-probe KD는 logit 방향성은 끌어올리나, on-distribution 데이터(제약상 금지)
  없이는 fluency를 못 만든다. 이미 from-scratch 학습 중인 Aster를 data-free KD가 이길 수 없다는
  사전 분석을 **실제 Aster에서 실측으로 확증**. 사용자 질문에 대한 완전한 답.

### Alternatives (대안)

- **toy proxy(TinyLM)로 천장만 측정**: 거부 — 사용자가 "실제 Aster 재현"을 명시 선택.
- **slm-cli chat으로 parity 검증**: 거부 — top-k=40 기본 샘플링으로 greedy 비교 불가, generate 사용.
- **ferry.build_vocab_map 그대로 사용**: 거부 — 49GB dense 행렬, SparseVocabMap으로 대체.
- **더 큰 steps/batch로 fluency 추구**: 거부(현 단계) — 천장은 step 수가 아니라 off-distribution
  probe·용량·basis가 결정. on-distribution 데이터가 진짜 해법이나 data-free 제약 위반(별도 승인 필요).

<!--
새 결정은 맨 위에 추가 (최신순).
MADR-lite 4섹션: Context / Decision / Consequences / Alternatives.
Status: Active | Superseded by DEC-NNN | Deprecated.
-->

## DEC-010 — tokenizer 문자열 기반 embed vocab-map 전이 (`--embed-vocab-map`)로 embed=0 collapse 타파

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] DEC-009에서 **embed zero-skip이 추론을 degenerate 고정점으로 붕괴**시킴을 실증
  (2B·9B 세 모델이 글자 단위 동일 word-salad). 의미 있는 추론엔 embed 전이가 선결 과제로 확인됨.
- [Fact] 막힘의 근본 원인은 vocab 불일치(256000≠48000)만이 아니라 **tokenizer 인코딩 방식 차이**:
  Aster=GPT-2 **byte-level BPE**(공백 `Ġ`, 한글은 raw-byte 문자), Gemma=**SentencePiece BPE**
  (공백 `▁`, 한글 그대로). raw 문자열 교집합은 12.8%(6166)뿐.
- [Fact] 데이터-free 제약 유지(probe·forward·KD 없음, tokenizer.json vocab 테이블만 디스크에서 읽음).

### Decision (결정사항)

1. **byte-level 정규화 후 매칭**: Aster 토큰을 GPT-2 char→byte 역매핑(`_byte_level_decoder`)으로
   실제 UTF-8로 디코드, Gemma는 `▁`→' ' 치환. 정규화 후 실제 교집합 **41.4%(19880, 한글 1826)**.
2. `build_vocab_map(student_tok, teacher_tok) → (t_for_s, stats)`: student id→teacher id 매핑
   (미매칭 -1). `transfer_embed`: teacher embed `(V_t,H_t)`를 **hidden축만** 우특이벡터로
   right-project `A V_n` (`H_t→H_s`), vocab축 V_t는 행 인덱스로 **보존**(양측 `_svd_project`는
   256000행을 rank로 압축해 대부분 토큰을 0으로 만드는 버그 → 전용 함수로 분리). 매칭 행만 scatter,
   미매칭 행은 zero 유지. 벡터화(index_select, Python 루프 제거).
3. `--embed-vocab-map` opt-in 플래그. 출력 디렉토리 `-embedmap` 접미사. 기본(off)은 DEC-008 force_skip 유지.
4. `print_report`/docstring을 모드별로 **정직 분기**: embed-map 시 seeded 비율 + basis-misalignment
   caveat 명시, off 시 기존 force-Skip 설명.

### Consequences (영향)

- [정직] **embed=0 collapse 타파 실증**: 2B embed-map chat 출력이 plain 2B(embed=0)와 완전히
  다름(영문 우세 word-salad vs 한영 혼합). 전이된 토큰 임베딩이 실제로 forward를 구동함을 확인.
- [정직한 한계] 여전히 word-salad — seeded embed의 hidden축이 **embed 자기 singular basis**라
  FFN/attn의 hidden 회전과 정렬되지 않음. weight tying하에서 logits geometry 불일치 잔존.
  collapse는 깼으나 fluent 전이는 아님(KD/학습 필요). 매칭 토큰 영문 편중(한글 1826/19880).
- 측정: coverage 0.9958→1.0, by_kind {VocabEmbed:1, CropPad:53, SvdProject:182},
  by_semantic partial 53→54·meaningless 105→104. schema PERFECT MATCH 유지(236, 0/0/0),
  embed nonzero 19880 rows, |max|=2.99. ferry 42/42 통과, py_compile OK.

### Alternatives (대안)

- **raw 문자열 매칭**: 거부 — 인코딩 차이로 12.8%만 겹침, 한글 0개. byte-level 정규화가 필수.
- **embed hidden축 CropPad(앞 1536)**: 가능하나 거부 — right-singular 투영이 embed 내부
  기하(토큰 간 상대거리) 에너지를 더 보존. 단 어느 쪽도 나머지 망과는 미정렬(공통 한계).
- **양측 `_svd_project` 재사용**: 거부 — vocab축(256000)을 rank로 압축해 대부분 토큰 0으로 붕괴(버그).
- **embed 매칭 없이 KD로 직행**: 거부(현 단계) — 데이터-free 제약 + KD는 별도 사용자 승인 필요.
## DEC-009 — 양측 직교 SVD 투영(`U_m^T A V_n`)으로 교체 + 9B 전이 실험으로 "큰 teacher = 불리" 실증

- **Date**: 2026-06-12
- **Status**: Active

### Context (배경)

- [Fact] 사용자가 9B teacher 실험을 요청하며, 기존 `ferry._svd_project`가 **진짜 SVD가 아님**을
  지적("AA이지 SVD가 아니였을거같은데"). 실제로 구버전은 `svd(A)`로 top-k 재구성한 뒤
  결과 행렬의 좌상단 블록을 슬라이스 — U·V를 인덱스로 잘라 **직교성을 깨고 선행 행/열에 편향**.
- [Fact] 레이어 수 불일치: gemma-2-9b L42 vs aster-1b L26. 사용자가 매핑 전략으로
  **uniform + front 둘 다 생성·비교**를 택함.
- [Fact] 사용자 의도(verbatim): "저차원 근사이지만 SVD보다 가능성 높은 알고리즘 채택".

### Decision (결정사항)

1. `_svd_project`를 **양측 직교 SVD 투영** `projected = U_m^T A V_n`로 교체.
   `u,_,vh = svd(A, full_matrices=False)`; `rank=min(M,N)`; `m=min(out_rows,rank)`,
   `n=min(out_cols,rank)` (rank 캡 필수); shape 불일치 시 항상 `_crop_pad`로 dst 정규화
   (넓은 축 zero-pad = teacher 신호 없음). Eckart-Young 최적 양측 저차원 제한.
2. `_svd_energy_kept`를 top-k 특이값 상한이 아닌 **실제 투영 후 Frobenius 비율**로 교체
   (두 축 동시 축소 시 top-k 상한은 과대평가).
3. 레이어 선택 매핑(`select_teacher_layers`, `--layer-select {uniform,front}`) 추가.

### Consequences (영향)

- [정직] **teacher가 클수록 순수 weight transfer는 불리**: FFN 에너지 보존
  2B=87.81% vs 9B=70.16%(uniform)/70.47%(front). 축소율(hidden 3584→1536, ffn 14336→6144)이
  지배 변수, 레이어 선택 전략은 미미(uniform vs front 0.3%p). "큰 teacher가 낫다"는 직관과 반대.
- [정직] **embed zero-skip이 추론을 degenerate 고정점으로 붕괴**: 2B·9B-uniform·9B-front 세 모델이
  chat greedy에서 글자 단위로 동일한 word-salad. embed `|max|=0`(tied lm_head도 0)이라 FFN
  전이 품질 차이가 추론으로 전혀 드러나지 않음 → 의미 있는 추론엔 embed 전이가 선결 과제.
- ferry 테스트 42/42 통과 (distill 회귀는 step 800→1500로 해소; 0.8 bar 유지 — 양측 투영의
  도달점이 더 높고 수렴만 느린 것을 5-seed로 규명, 임계 낮추기 아님).

### Alternatives (대안)

- **구버전 슬라이스 유지**: 거부 — 직교성 파괴·편향, 사용자 지적대로 진짜 SVD 아님.
- **distill 임계 0.8→0.76 하향**: 거부 — 증상 은폐. 도달점이 더 높음을 측정으로 확인하고 step만 증가.
- **front만 / uniform만**: 거부 — 사용자가 둘 다 비교 요청, 차이 미미함을 실증으로 남김.

## DEC-008 — Gemma-2-2B → Aster aster-1b 순수 가중치 초기 전이 (KD 없음, 정직한 negative-result PoC)

- **Date**: 2026-06-11
- **Status**: Active

### Context (배경)

- [Fact] 사용자가 별도 Rust 프로젝트 `../SLM_FROM_BEGIN`의 from-scratch SLM **Aster**를 사전학습
  LLM으로부터 **초기 전이**할 수 있는지 물었다. 질문 연쇄 끝에 범위를 확정: **KD 제외**,
  **teacher = Gemma 계열**, **순수 weight-space 전이만**(Ferry Stage1 식 algebra), **GPU 불필요**,
  출력은 `./test_output`, 스크립트는 SLM_CLONER 폴더. teacher는 "**2B 먼저**"로 시작.
- [Fact] Aster `aster-1b`(target): d_model 1536 / L26 / heads16·kv8 / head_dim96 / ffn 6144 /
  vocab 48000. RMSNorm, weight-tying, GeGLU(gelu) FFN. **step 3600 학습 진행 중** — 덮어쓰기 금지.
  텐서 네임스페이스 `v2.*` (embed/blocks.N.{q,k,v,o,ffn_gate,ffn_up,ffn_down,attn_norm,ffn_norm}/
  final_norm), 236 텐서.
- [Fact] `google/gemma-2-2b`(teacher, 실측): vocab 256000 / hidden 2304 / **L26** / heads8·kv4 /
  head_dim256 / ffn 9216 / gelu_pytorch_tanh(GeGLU) / soft-cap 50·30 / RMSNorm. **레이어 수 26이
  aster-1b와 정확히 일치**, soft-cap·activation family도 일치 → 9B보다 깔끔한 매핑. 다운로드 ~5GB.
- [Fact] 비교 불가 축: vocab(256000 vs 48000, **다른 토크나이저** → 토큰 id 의미 불일치),
  head_dim(256 vs 96) + RoPE base 불일치 → attention 기하 전이 무의미. hidden·ffn은 SVD 축소 가능.

### Decision (결정사항)

독립 스크립트 `transfer_gemma_to_aster.py`를 추가한다. **데이터-free·training-free·probe 없음**,
오직 `ferry.transform_tensor`(Copy/CropPad/SvdProject/Skip)의 결정론적 선형대수만 사용.

- **이름 매핑**(`MAP_RULES`): gemma-2 `model.*` ↔ aster `v2.*`, 26 레이어 1:1. semantic 태그로
  정직성 분리 — `meaningful`(ffn gate/up/down, GeGLU 양쪽 일치) / `partial`(norm gamma, CropPad) /
  `meaningless`(embed vocab 불일치, attn head_dim+RoPE 불일치).
- **embed는 `force_skip=True`** — 둘 다 2D라 SVD가 적용되겠지만 vocab 축이 비교 불가능하므로
  의도적으로 zero-init 유지(노이즈 주입 거부). 이것이 정직한 "신호 없음" 표현.
- **정직 지표 = SVD 스펙트럼 에너지 보존율**(`_svd_energy_kept`). zero-init 대비 weight drift는
  항상 1.0이라 무의미하므로 폐기하고, "teacher 행렬의 제곱 특이값 중 top-k가 살아남은 비율"로 교체.
- 출력은 **신규 파일만**: `./test_output/aster-1b-from-gemma/{params.safetensors, transfer_report.json}`.
  live aster-1b 체크포인트는 절대 미접촉.

### Consequences (영향)

- [측정] coverage 0.9958(embed 1개만 Skip), by_kind {Skip:1, CropPad:53, SvdProject:182}.
  **FFN SVD 에너지 보존 87.8%**(meaningful 78텐서) — hidden·ffn 축소에도 teacher 스펙트럼 88% 생존.
  attention은 에너지 97.95% 보존되지만 semantic=meaningless로 분리 → **높은 에너지 ≠ 기능적 의미**를
  수치로 정직하게 드러냄.
- [검증] 출력 236 텐서 이름+shape이 **실제 aster-1b params.safetensors와 PERFECT MATCH**
  (missing/extra/mismatch 0) → Aster Rust 런타임이 실제 로드 가능한 형식임을 실측 확인.
  embed는 all-zero(forced Skip), 전부 float32, ~3.98GB.
- [한계, 정직] 이것은 **작동하는 모델이 아니라 초기 skeleton**이다. KD/추가학습 없이는 Gemma 품질은
  커녕 유창한 출력도 불가. attention·embed 무의미, FFN만 부분 의미. negative-result를 숨기지 않고
  수치(ffn_svd_energy_kept, semantic별 분리)로 보고하는 것이 이 PoC의 목적.

### Alternatives (대안)

- **KD/증류 추가**: 거부됨(사용자가 KD 제외 명시). 데이터·forward·gradient 필요 → 범위 밖.
- **더 큰 teacher(Gemma-2-9B/Gemma-4)**: 순수 weight transfer에서는 teacher가 클수록 SVD 축소율↑로
  불리. Gemma-4 계열은 per-layer-input·kv-shared 등 비표준 구조라 매핑이 깨짐. 2B가 L26 일치로 최적.
- **embed를 SVD 투영**: 거부됨. vocab 축 비교 불가 → 노이즈 주입일 뿐. force_skip이 정직.
- **weight drift 지표 유지**: 거부됨. zero-init 대비 항상 1.0 → 무정보. SVD 에너지 보존율로 교체.

## DEC-007 — 실제 Qwen3-0.6B teacher → ferry-?B(아키텍처 변경) student, CPU·데이터-free 증류

- **Date**: 2026-06-11
- **Status**: Active

### Context (배경)

- [Fact] 사용자가 `ferry_qwen.py`에서 **실제 Qwen3-0.6B를 증류**해 더 작은 `ferry-?B`로 만들고
  결과를 테스트할 수 있게 해달라고 요청했다. 이어 "**아키텍처를 변경해도 동일한지가 중요함**"과
  "**GPU는 사용하지 말 것**"을 못박았다.
- [Fact] 환경: `transformers`/`accelerate` 미설치였고 설치 승인받음(현재 transformers 5.10.2,
  accelerate 1.13.0). Qwen3-0.6B는 캐시에 없었고 HF 네트워크 200 → 다운로드 승인받음.
- [Fact] Qwen3-0.6B config: vocab 151936, hidden 1024, intermediate 3072, layers 28,
  heads 16/KV 8, head_dim 128, **tied embeddings**, silu, RMSNorm. 총 596M 파라미터.
- [Fact] CPU 검증: fp32 로드 ~39s, forward(1×8) 0.43s. student(h512/L8/heads4·kv2, 103M)와
  teacher의 state_dict **이름 91/91 공유, exact-shape는 16개뿐** → 나머지는 shape 불일치.

### Decision (결정사항)

`ferry_qwen.py`를 추가한다. teacher는 실제 `Qwen/Qwen3-0.6B`(transformers, **CPU 전용**,
`dtype=torch.float32`, `CUDA_VISIBLE_DEVICES=""`/`device_map` 미사용). student `ferry-?B`는
**같은 `Qwen3ForCausalLM` 클래스이지만 깊이·폭이 다른** 더 작은 config(기본 ferry-0.1B:
hidden 512, intermediate 1536, layers 8, heads 4/KV 2, head_dim 128, tied) — vocab/tokenizer
동일하므로 VocabMap 불필요. Ferry 파이프라인 재사용: `ferry.transfer`(이름 매칭 → Copy/CropPad/
SvdProject/Skip)로 warm-start 후 `ferry.distill`을 **합성 토큰 probe**(random token ids,
데이터-free)로 실행해 logits MSE를 맞춘다. 결과는 per-token agreement + greedy 생성으로 테스트.
CPU·시간 제약 때문에 batch/seq/steps는 작게 둔다.

### Consequences (영향)

- Ferry의 "아키텍처가 달라도 같은 답" 주장을 **장난감(TinyLM)이 아닌 실제 0.6B 모델**에서 시연.
- 새 의존성 2개(transformers, accelerate)와 ~1.2GB 모델 다운로드가 프로젝트에 들어옴 →
  단일파일/최소의존 원칙의 예외(이 파일에 한정, 사용자 승인).
- **데이터-free 유지**: 합성 토큰 probe만 사용(코퍼스/디스크 없음). 단, random 토큰은
  off-distribution이라 자연어 품질 보장이 아니라 *logit 정합 PoC*임을 정직하게 명시(정직한 한계).
- GPU 금지 → 큰 vocab(151936) logits MSE를 CPU에서 다루므로 느림. PoC 규모로 제한.
- 테스트는 deps/model 없으면 `pytest.importorskip`/skip으로 게이트(다른 42개 테스트와 분리).

### Alternatives (대안)

- student로 ferry.py의 TinyLM 재사용: 다운로드 작지만 Qwen과 아키텍처가 달라 transfer 대부분
  무의미, 사용자의 "아키텍처 변경해도 동일" 요구와 약하게 부합 → 기각.
- 실제 텍스트 probe 허용: 자연어 품질엔 유리하나 **데이터-free 하드 제약(DEC-001) 위반** → 기각.
- GPU 사용: 빠르지만 사용자가 명시적으로 금지 → 기각.

## DEC-006 — Stage 0 어휘 정합(vocabulary reconciliation) 도입

- **Date**: 2026-06-11
- **Status**: Active

### Context (배경)

- [Fact] LLM의 출력 축은 **그 모델 자신의 vocabulary**다. teacher/student가 다른 tokenizer로
  만들어지면 LM-head 너비가 다르고(예: V_t=64 vs V_s=48), 같은 토큰 id `j`가 서로 다른
  토큰을 가리킨다.
- [Fact] 기존 LLM 경로(stage 1–3)는 `teacher.vocab == student.vocab`을 암묵 가정해, 다른
  어휘에서는 `align_output`/`distill`이 shape error를 내거나 무관한 토큰 열을 비교했다.
- [Fact] 사용자가 "병합 전 vocabulary를 student에 맞추는 과정이 추가되어야 한다, 이것도
  적용하자"로 stage 0을 명시 요청했다.

### Decision (결정사항)

병합 전 단계로 `reconcile_vocab(student, teacher, t_for_s=None) → VocabMap`을 추가한다.
`VocabMap`은 `t_for_s`(student id→teacher id, `-1`=student 전용)와 `(V_t×V_s)` 선택 행렬
`projection`을 들고, `remap_ids`(probe를 teacher id로 번역)와 `project`(teacher logits를
student 어휘 공간으로 사상) 두 연산을 제공한다. `agreement`/`align_output`/`distill`은
선택적 `vocab_map=None` 인자를 받으며 `None`은 엄격한 no-op(기존 동작 보존)이다. 기본
`t_for_s`는 shared-prefix 맵이고, 실제 배포는 tokenizer 문자열 매칭 맵을 넘긴다.

### Consequences (영향)

- 다른 vocab의 teacher/student도 정합·distill 가능: 데모(V 64→48) per-token
  0.410 → 0.706(align) → 0.904(distill).
- student 전용 토큰(`-1`)은 teacher 신호가 없어 0 타깃 열 → 그 토큰들은 원리상 정합 불가
  (정직한 잔차).
- `distill`은 token 모드에서만 `vocab_map`을 허용(continuous 모드에 넘기면 ValueError).
- 기존 동일-vocab MLP/TinyLM 동작·테스트는 no-op 경로로 그대로 유지.

### Alternatives (대안)

- text-level KD(생성 문자열 비교): tokenizer 디코딩·문자열 매칭 필요, 데이터-free·closed-form
  원칙과 충돌 → 기각.
- 공유 임베딩 공간 학습: 추가 학습·복잡도 증가, PoC 단순성 위배 → 기각.

## DEC-005 — gradient distillation(Stage 3) 도입, no-gradient 제약 해제

- **Date**: 2026-06-10
- **Status**: Active

### Context (배경)

- [Fact] 닫힌 형식 정합(stage 1–2, stage 2b)은 비선형 teacher를 ~0.97까지만 닫고,
  `TinyLM` 자기회귀 생성은 잔차가 누적해 token-match가 .52→.34로 감쇠했다.
- [Fact] 한계 분석(DEC-003 후속) 결과: 이 잔차는 함수의 한계가 아니라 **닫힌 형식 방법의
  한계**였다 — 보편 근사상 충분한 student는 표현 가능하나 `lstsq`가 그 해에 닿지 못함.
- [Fact] 사용자가 "코드까지 바꿔보자, 제약은 다 풀어줄게"로 기존 no-gradient hard
  constraint를 **명시적으로 해제**했다.
- [Risk] gradient loop이 데이터-free 원칙을 흔들 수 있음 → 합성 probe로 제한해 차단.

### Decision (결정사항)

Stage 3 `distill`을 추가한다. Adam으로 `mse_loss(student(probe), teacher(probe))`를
최소화하되, **매 step 새 합성 probe**(continuous=`synthetic_probe`, token=`token_probe`)를
뽑고 teacher 출력을 타깃으로 쓴다. 닫힌 형식 stage 1–2를 warm start로 선행한다. 입력 모드는
`in_dim` XOR (`vocab`+`seq`)로 정확히 하나만 허용. 데이터셋·디스크 I/O는 계속 금지.

### Consequences (영향)

- 비선형 depth-matched ActMLP held-out top-1 ~0.99(.996/.992/.996), `TinyLM` per-token
  0.41→0.89, 생성 step8 .34→.61로 한계가 닫힘/평탄화.
- AGENTS.md hard-constraints의 "no gradient loop" 항목을 "gradient allowed, stage 3 한정,
  data-free"로 교체. theory.html §9 신설, 결론 §10으로 이동.
- 테스트 28→32(distill 4종). 자기회귀 잔차는 평탄화일 뿐 horizon 증가分은 잔존(정직한 한계).
- 롤백 비용: `distill` 함수 + 데모/테스트/문서 일부 제거면 stage 1–2 PoC로 복귀 가능.

### Alternatives (대안)

- 닫힌 형식 random-feature 기저 확장: 고정 probe 과적합으로 held-out 미개선(실험으로 기각).
- 실제 데이터 KD: data-free 원칙 위반으로 기각.
- 폭만 키우기: depth mismatch엔 ~0.86에서 정체, 단독으로는 불충분(기각).

## DEC-004 — TODO↔Issue 연동 정책 off

- **Date**: 2026-06-10
- **Status**: Active

### Context (배경)

- [Fact] 프로젝트는 git 저장소가 아니고 단일 파일 PoC다.
- [Fact] `/init-docs`는 문서 생성 전 TODO↔Issue 정책(strict|balanced|off) 결정을 요구한다.

### Decision (결정사항)

정책을 **off**로 설정. 이슈 연동은 제안만 하고 강제하지 않는다. `AGENTS.md` 운영 규칙에 기록.

### Consequences (영향)

- TODO 추가/완료 시 `/ticket` 강제 없음. 로컬 단일 개발자 흐름에 가벼움.
- 추후 git 저장소화 + 협업 시 balanced/strict로 승격 가능.

### Alternatives (대안)

- balanced/strict: 협업·추적 강화하나 git 없는 현 상태에 과함. 기각.

## DEC-003 — same-answer 보장 목표로 pivot + synthetic probe 허용

- **Date**: 2026-06-10
- **Status**: Active

### Context (배경)

- [Fact] 사용자 목표(m0052): "Teacher, Student 모델이 같은 답을 내도록 보장".
- [Risk] 기존 "no behavioral probing" 제약과 "same-answer 보장" 목표가 충돌.

### Decision (결정사항)

목표를 same-answer 보장으로 확정. 충돌 해소를 위해 **synthetic probe(랜덤 텐서) 허용**, 단 외부/실제 데이터셋은 계속 금지(m0054). Stage 2 `align_output`을 closed-form `lstsq`로 추가.

### Consequences (영향)

- `synthetic_probe`/`agreement`/`align_output` 추가. same-answer는 **조건부**(rank condition) 보장으로 정의됨.
- `AGENTS.md` Hard constraints에서 no-probing 규칙을 synthetic-probe-allowed로 supersede.

### Alternatives (대안)

- 데이터셋 사용: hard constraint 위반, 기각.
- gradient KD loop: no-training-loop 제약 위반, 기각.

## DEC-002 — 단일 파일 구조 + import alias 유지

- **Date**: 2026-06-10
- **Status**: Active

### Context (배경)

- [Fact] 사용자 요구(m0016): "내가 수정할 수 있게 최대한 단순하게".
- [Fact] 제품명 Ferry 확정(m0029)으로 `clone.py`→`ferry.py` 리네임 발생.

### Decision (결정사항)

torch만 쓰는 **단일 파일** `ferry.py` + `test_ferry.py`. packaging/CLI/멀티모듈 금지. 테스트는 `import ferry as clone` alias로 본문은 `clone.*` 유지.

### Consequences (영향)

- 편집 용이. 단, 파일 리네임 시 alias 라인도 갱신 필요(함정).

### Alternatives (대안)

- `src/slm_cloner/` 멀티모듈 레이아웃: 단순성 요구에 반함, 기각.

## DEC-001 — weight-space 결정론적 전이 (데이터 없음)

- **Date**: 2026-06-10
- **Status**: Active

### Context (배경)

- [Fact] 사용자 제약(m0010): 학습 데이터 금지, 모델 간 전이만.
- [Fact] 초기 adapter+KD(데이터 필요) 계획은 이 제약으로 무효화.

### Decision (결정사항)

Stage 1을 이름 기반 매칭 + 결정론적 텐서 변환(Copy / CropPad / SvdProject / Skip)으로 구현. 새 변환은 `transform_tensor` 분기만 확장.

### Consequences (영향)

- 데이터/디스크 I/O 불필요. shape mismatch는 crop/pad/SVD로 흡수, rank mismatch는 Skip(학생 유지).
- 구조가 다른 모델 간 same-answer는 Stage 1만으로는 불가 → Stage 2 필요(DEC-003).

### Alternatives (대안)

- Adapter+KD with training data: 데이터 제약 위반, 기각.
- permutation alignment: 복잡도↑, 미래 작업으로 연기.
