# Ferry — Session Handoff

> 마지막 갱신: 2026-06-12 (공개 git repo 발행 `WeightForge` + strict source-available LICENSE, DEC-015; 그 전 Aster 실모델 확장 아크 DEC-008~014). 세션 간 인수인계용 요약. 권위 있는 세부는 `.agents/` 6 docs와 `AGENTS.md`를 따른다.

## 1. 한 줄 요약

**Ferry** = layer 수 / hidden dim / vocabulary가 다른 Teacher↔Student 모델이 *같은 답*을 내도록, **학습 데이터 없이**(합성 probe만) (0) 어휘 정합 + (1) weight-space 전이 + (2) closed-form 출력·은닉 정렬 + (3) gradient distillation으로 맞추는 PoC. same-answer는 **조건부(rank condition)** 로만 보장되며, 안 되는 경우 residual을 정직히 보고한다(가짜 일치 없음).

**확장(2026-06-11~12)**: 같은 Ferry 파이프라인을 **실모델 Gemma-2 → Aster(`../SLM_FROM_BEGIN`의 Rust from-scratch SLM)** 에 적용 (DEC-008~014, §10). 순수 전이 → embed collapse 타파 → 한글 reachability → closed-form 정렬 → 한글 가중 정렬까지 진행. 전부 CPU·data-free·신규 파일만 출력(live 학습 미접촉). 정직 결론: **한글 도달은 달성, 유창성은 학습/KD 영역**.

## 2. 파일 구조 (flat, 패키징 없음)

> **공개됨**(DEC-015): `github.com/p4r4d0xb0x/WeightForge`(PUBLIC, 코드네임 Ferry). 초기 커밋 `770bfece`,
> `origin/main` 푸시 완료. `test_output/`(23GB)는 `.gitignore`로 제외(git 미진입). data-free 불변.

| 파일 | 역할 |
|---|---|
| `ferry.py` | 전체 로직 + 6-part toy demo. 모델 `MLP`(선형·정확 보장) / `ActMLP`(비선형) / `TinyLM`(트랜스포머) |
| `test_ferry.py` | pytest **42 cases**. **`import ferry as clone`** alias 사용(본문 전부 `clone.*`) |
| `ferry_qwen.py` | 실모델 확장(DEC-007): 실제 Qwen3-0.6B → 아키텍처 변경 `ferry-?B`. CPU 전용·데이터-free |
| `test_ferry_qwen.py` | gated 테스트 6종(`pytest.importorskip('transformers')`, 모델 없으면 skip) |
| `transfer_gemma_to_aster.py` | **Aster 확장**(DEC-008/009/010/012): Gemma-2 → aster-1b 순수 weight 전이 + embed vocab-map + byte-composition. ~840 LOC |
| `ferry_aster.py` | **Aster 확장**(DEC-011/013): Aster PyTorch 재현(parity 바이트 일치) + Gemma-2B data-free KD + `final_hidden()`. 479+ LOC |
| `align_aster_embed.py` | **Aster 확장**(DEC-013 b′ + DEC-014): closed-form 직교 Procrustes embed-basis 정렬 + 한글 가중(`--kr-weight`). ~300 LOC |
| `test_output/` | Aster 산출물(전이/KD/정렬 params·report). git 미추적 |
| `theory.html` | self-contained 이론 문서(공학자 판본, 무의존성). 시각 번호 §0–11, svg 8, table 7 (toy core 한정) |
| `README.md` | **공개 repo 최소 README**(DEC-015) — 요약·4-stage·layout·실행법·정직한 한계·LICENSE 안내 |
| `LICENSE` | **strict source-available·publication-reserved**(DEC-015). §2 좁은 use, §3 금지 9종(타 모델 train/distill·patent·competing 포함), OSI 비승인 |
| `.gitignore` | `test_output/`(23GB)·모델 가중치·파이썬/테스트 캐시·venv·OS junk 제외(DEC-015) |
| `AGENTS.md` | 프로젝트 규약 — what/goal/hard constraints/layout/commands/core pipeline/gotchas |
| `.agents/*.md` | GOAL/PLAN/TODO/PROGRESS/DECISION/MEMORY 6 docs |
| `handoff.md` | 이 문서 |

## 3. 핵심 파이프라인 (`ferry.py`)

- **Stage 0 — 어휘 정합(LLM 전용)**: `reconcile_vocab` → `VocabMap`(`t_for_s` student→teacher id, `-1`=student 전용 + `(V_t×V_s)` 선택 행렬 `projection`). `remap_ids`(probe→teacher id) / `project`(teacher logits→student 공간). `agreement`/`align_output`/`distill`에 `vocab_map=None` 옵션(None=엄격 no-op). 기본 맵=shared-prefix(데모용), 배포는 tokenizer 매칭 맵 필요.
- **Stage 1 — weight transfer**: `extract_spec → match_tensors(이름 기반) → transform_tensor → transfer → report`. 변환 4종 `Copy`/`CropPad`/`SvdProject`/`Skip`. 새 변환은 `transform_tensor` 분기만 확장.
- **Stage 2 — output alignment**: `align_output`이 마지막 `nn.Linear`에 forward hook → `[F|1] @ W = teacher(probe)`를 `lstsq` 1회 solve → weight/bias write-back. `_flatten_logits`로 `(n,seq,vocab)`도 처리(2D는 no-op).
- **Stage 2b — hidden alignment**: `align_hidden` forward sweep으로 student 각 은닉 linear의 pre-activation을 teacher 대응에 `lstsq` 회귀 후 head 정합. **flat MLP 계열만**(`TinyLM`은 head-only fallback). gradient 없음.
- **Stage 3 — gradient distillation**: `distill` warm-start 후 Adam, **매 step 새 합성 probe**로 `mse_loss(student, teacher)` 최소화. 데이터-free. 입력 모드 `in_dim` XOR (`vocab`+`seq`). gradient는 stage 3에만 격리.

## 4. 증명한 정리 (the point)

closed-form `align_output`은 teacher 출력맵이 student 마지막 특징의 아핀 함수일 때(**rank condition**, 실무적으로 student penultimate width ≥ teacher 출력맵 rank)에 한해 `student(x)==teacher(x)`를 **모든 x**에 보장.
- 선형 teacher 충분 시: held-out `top1_agree=1.0`, `mse≈8e-15`.
- 비선형 teacher: head-only 부족 → `align_hidden` ~0.97(깊이 일치) → `distill` ~0.99.
- capacity sweep(held-out top-1, deeper teacher `MLP[32,128,96,64,10]`): w4=.471 / w8=.467 / w16=.562 / **w48=1.000**. 좁으면 bottleneck = honest limit(버그 아님).

## 5. Hard constraints (위반 금지)

- 학습 데이터 / 데이터셋 / 데이터 로더 / 데이터용 disk I/O **금지**(데이터-free 유지).
- **synthetic probe 허용**(DEC-003): `synthetic_probe`/`token_probe`/`shared_token_probe`. 실제/로드 데이터 금지.
- **gradient training 허용**(DEC-005, 사용자가 no-gradient 해제) — 단 stage 3 `distill`에만, 데이터-free. stage 1–2는 결정적 closed-form.
- **GPU 금지**(DEC-007): `ferry_qwen.py`는 `CUDA_VISIBLE_DEVICES=""`·CPU·float32.

## 6. 실모델 확장 (`ferry_qwen.py`, DEC-007)

teacher 실제 `Qwen/Qwen3-0.6B`(596M), student `ferry-0.1B`(103M=17.3%·절반 깊이) 등 **아키텍처 변경**. `LogitsModel` 어댑터로 HF `CausalLMOutput`→raw logits 변환해 `ferry.transfer`/`distill` 재사용. same-vocab → VocabMap 불필요. REAL: transfer 91/91 coverage 1.0, per-token top1 **0.000→0.195**, **plateau ~0.19~0.21**(정직한 PoC ceiling: 17% 파라미터·151936 vocab diffuse MSE·off-distribution probe). 의존성: `transformers` 5.10.2 + `accelerate` 1.13.0 설치됨, Qwen 캐시됨(~1.2GB).

## 7. 주요 함정 (gotchas)

- **import alias**: `test_ferry.py`의 `import ferry as clone` — `ferry.py` 리네임 시 갱신.
- `report()`의 `mean_relative_error`는 Stage-1 weight 드리프트 지표, 답변 품질 아님.
- same-answer는 width 조건부. bottleneck plateau 정상.
- 비선형은 `align_hidden` 필요(얕은 student는 부분 한계). 자기회귀 잔차는 distill로 **평탄화되나 소거 아님**.
- 다른 vocab은 Stage 0 필수. 세 축 동시 불일치가 현실적 최악(`_demo_combined_mismatch`).
- **Qwen3 bfloat16 기본** → `.float()` 강제 안 하면 CPU autograd backward 에러.
- capacity·demo 수치는 `theory.html`에 하드코딩 — 데모 teacher 변경 시 동기화.

## 8. 검증

```bash
python -m pytest test_ferry.py -q        # 42 passed
python ferry.py                          # 6-part demo
python -m pytest test_ferry_qwen.py -q   # 6 passed (모델 없으면 skip)
python ferry_qwen.py                     # 실모델 증류 리포트(CPU)
```

상태: 모든 테스트 통과, demo 정상, theory.html 구조 유효(section 12/12·svg 8·table 7·번호 0–11). `todo!()`/`TODO`/skipped test 없음(`Skip`은 정당한 변환 kind).

## 9. 다음 작업 (정찰로 식별, 미착수)

- `ferry_advance.py` — 더 작은 student용 activation-aware 저차원 전이(DEFERRED, 프로토타입 검증됨).
- tokenizer 문자열 기반 `t_for_s` 빌더(stage-0 배포용).
- permutation 정렬(Git Re-Basin류) / 트랜스포머 전층·attention 정합 — 자기회귀 잔차 근본 축소.
- Qwen plateau 완화 실험, capacity sweep JSON export, non-MLP toy 데모.

상세·우선순위: `.agents/TODO.md`, `.agents/PROGRESS.md`, `.agents/DECISION.md` 참조.

## 10. Aster 실모델 확장 (Gemma-2 → Aster, DEC-008~014)

같은 Ferry 파이프라인을 **실모델**에 적용. 대상 student = `../SLM_FROM_BEGIN`의 Rust from-scratch SLM **aster-1b**(d1536/L26/h16·kv8/head_dim96/ffn6144/vocab48000, tied embed, safetensors 236 텐서 `v2.*`). teacher = `google/gemma-2-2b`(KD·정렬), `google/gemma-2-9b`(전이 실험). **전부 CPU·data-free·신규 파일만**(live aster-1b ~step3950+ 학습 미접촉).

### 진행 위계 (각 단계가 앞 단계의 한계를 드러냄)

| DEC | 산출물 | 한 것 | 정직한 한계 |
|---|---|---|---|
| 008 | `transfer_gemma_to_aster.py` | 순수 weight 전이(Stage1 algebra). coverage 0.9958, FFN energy 87.8%, schema PERFECT MATCH | 작동 모델 아닌 **초기 skeleton**. embed/attn 무의미 |
| 009 | `ferry._svd_project` 교체 + 9B | 양측 직교 SVD `U_mᵀ A V_n`(Eckart-Young) + 9B 실험 | **big teacher 불리**(FFN 2B 87.8% vs 9B 70.2%). **embed=0 collapse** 발견 |
| 010 | `--embed-vocab-map` | tokenizer 문자열 매칭 embed 전이(41.4%=19880). **collapse 타파**(2B≠9B 출력) | 여전히 word-salad. 한글 1826개뿐 |
| 011 | `ferry_aster.py` | Aster PyTorch 재현(parity 바이트 일치) + Gemma-2B data-free KD | cosine -0.08→0.77이나 **top1=0 천장**(off-distribution probe) |
| 012 | `--embed-byte-compose` | byte-fallback MEAN seed. 한글 reachability: mass 2%→91%, 샘플링 0/30→30/30, 커버리지 47080 | **diffuse**(top1 못 됨), byte-order 손실, 비유창 |
| 013 | `align_aster_embed.py` (b′) | closed-form 직교 Procrustes embed-basis 정렬. best_rank 95→9, top-k40 한글 생존 1/10→8/10 | 2/10 회귀, generic, greedy top1 비한글, **유창성 미달** |
| 014 | `align_aster_embed.py --kr-weight 50` | 한글 앵커 50× 업웨이트 가중 Procrustes. median rank **9→2**, 첫 greedy-Korean, top-k40 한글 다수 | 2/10 회귀 **잔존**, greedy 대부분 `<pad>`, generic·비유창 |

### 핵심 개념: reachability vs alignment

- **도달(reachability) ✅**: byte-comp이 한글 row를 비영 벡터로 → softmax mass 91%. b′ 정렬이 de-diffuse → top-k 생존.
- **정렬(alignment) ❌**: embed 방향이 Aster hidden 기하·서로와 미정렬 → diffuse·비유창. 깨진 3지점: (1) hidden basis 불일치(closed-form 정렬로 부분 회복), (2) byte-order 손실, (3) 문법 신호 부재. **(2)(3)은 학습/KD 영역** — data-free closed-form으로 불가.
- **Rust에서 한글 보기**: 정렬 전 `--top-k 0` 필수, 정렬 후 기본 `--top-k 40`에서 생존.

### Aster 확장 함정 (필수)

- **forward 재현 정밀**: embed NO sqrt scaling / RMSNorm **raw gamma**(Gemma `(1+gamma)` 아님) / RoPE **interleaved GPT-J**(rotate_half 아님, dim96 full) / GQA `kv_head=h//2` / scale `1/sqrt96` / softcap 50·30 / GeGLU **gelu-tanh** / tied embed / weight `[out,in]`. 하나라도 틀리면 parity 깨짐.
- **parity는 `generate --top-k 0 --temperature 0.0 --repetition-penalty 1.0`(순수 greedy)로만** 검증(`chat`은 top-k40 부가처리라 부적합).
- **bfloat16 함정**: Gemma teacher/student `.float()` 강제(CPU autograd backward 에러 회피).
- **`build_vocab_map` 미변경 유지**(KD logits SSOT + 재현성). byte-comp은 embed seeding 전용 additive.
- **`SparseVocabMap` 사용**(`ferry.build_vocab_map`의 dense 49GB 회피).
- **`--iters` 기본 1 고정**(정렬 과반복 시 anti-Korean prior 과회전으로 한글 붕괴).
- **`--kr-weight` 가중은 Procrustes 단계에만**(한글 앵커 9% 업웨이트). `e_fit_whole` lstsq target은 가중 무관(앵커별 독립 회귀)이라 건드리지 말 것. sweet spot=50(rank 9→2), default 1.0=DEC-013 재현.

### 재생 명령 (각 신규 파일, ./test_output 출력)

```bash
python transfer_gemma_to_aster.py --teacher google/gemma-2-2b --embed-vocab-map --embed-byte-compose --out ./test_output  # 전이+byte-comp
python ferry_aster.py --steps 100 --batch 8 --seq 16                # data-free KD
python align_aster_embed.py --n-batches 16 --iters 1                # b′ closed-form 정렬 (uniform)
python align_aster_embed.py --kr-weight 50 --iters 1 --n-batches 16 --out ./test_output/aster-1b-from-gemma-2-2b-embedmap-bc-aligned-krw50  # 한글 가중 정렬 (DEC-014)
# chat 실증(../SLM_FROM_BEGIN에서, 한글 보려면 --top-k 0):
printf '옛날 옛적에\n/exit\n' | ./target/release/slm-cli chat --model <dir> \
  --tokenizer /data/0A_DATASET/L0_LLM/V3/TOKENIZER/tokenizer.json \
  --model-config configs/model/pretrain-1b.toml --device cpu \
  --max-new-tokens 40 --temperature 1.0 --top-k 0
```

### 다음 (사용자 결정 대기)

- **on-distribution KD/학습** = meaning alignment의 유일한 진짜 해법이나 **data-free 제약 위반 → 별도 승인 필수**.
- byte-order 보존 seeding / 앵커 가중 Procrustes(회귀 완화) — data-free 유지 가능.
- 또는 PoC 종료. 상세는 `.agents/TODO.md` Medium·`DECISION.md` DEC-011/013 Alternatives.
