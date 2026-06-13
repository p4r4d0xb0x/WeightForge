---
project: Ferry
created: 2026-06-10
updated: 2026-06-13 (DEC-016 커밋 이메일 noreply 교체 + filter-branch 스크럽 함정 추가)
---

# Memory

Cross-session knowledge base. 반복 패턴, 교훈, 주의사항.

## Patterns

- **파이프라인**: Stage 0(`reconcile_vocab` 어휘 정합, LLM 전용) → Stage 1(weight-space algebra) → Stage 2(closed-form output alignment) + Stage 2b(`align_hidden` 은닉 정합) → Stage 3(`distill` gradient fine-tune). Stage 1은 초기화/근사, Stage 2/2b가 closed-form same-answer, Stage 3가 닫힌 형식이 못 닫는 비선형/깊이/자기회귀 한계를 닫는다.
- **Stage 0 `reconcile_vocab` (VocabMap)**: teacher/student vocab이 다르면 LM-head 너비·토큰 의미가 달라 직접 비교 불가. `VocabMap`은 `t_for_s`(student→teacher id, `-1`=student 전용) + `(V_t×V_s)` 선택 행렬 `projection`을 들고, `remap_ids`(probe→teacher id)·`project`(teacher logits→student 공간) 제공. `agreement`/`align_output`/`distill`에 `vocab_map=` 옵션으로 주입, `None`은 엄격한 no-op. 기본 맵은 shared-prefix(`arange(min(V_t,V_s))`), 배포는 tokenizer 매칭 맵 필요. `distill`은 token 모드에서만 허용(continuous면 ValueError).
- **Stage 3 `distill`**: warm start(stage 1–2) 후 Adam으로 `mse_loss(student(probe), teacher(probe))` 최소화. **매 step 새 합성 probe** 리샘플이 일반화의 핵심(고정 probe는 과적합). 입력 모드 `in_dim` XOR (`vocab`+`seq`). 데이터-free 유지(teacher가 타깃 제공). gradient는 stage 3에만 격리.
- **새 변환 추가는 `transform_tensor` 분기만**: Copy / CropPad / SvdProject / Skip. enum-like kind 문자열로 분기.
- **closed-form 정렬**: 마지막 `nn.Linear`에 forward hook → 입력 feature `F` 캡처 → `[F|1] @ W = teacher(probe)`를 `torch.linalg.lstsq`로 1회 solve → weight/bias write-back. gradient loop 없음.
- **Aster 확장 파이프라인 (실모델 Gemma→Aster, DEC-008~014)**: toy core와 별개 파일군. `transfer_gemma_to_aster.py`(순수 weight 전이 + embed vocab-map + byte-composition) → `ferry_aster.py`(Aster PyTorch 재현 + Gemma-2B data-free KD, parity 바이트 일치) → `align_aster_embed.py`(closed-form Procrustes embed-basis 정렬, b′ + Korean-weighted `--kr-weight`). 모두 CPU-only·data-free·신규 파일만 출력(live aster-1b 학습 미접촉). 산출물은 `./test_output/`. **결론 위계**: 순수 전이=초기 skeleton(작동X) → embed=0 collapse → vocab-map으로 collapse 타파 → byte-comp으로 한글 *reachability* → b′ 정렬로 *top-k 생존* → kr-weighted로 *rank 9→2*. **유창성(meaning alignment)은 closed-form으로 불가, 학습/KD 영역**.

## Heuristics

- same-answer 보장은 **조건부**로만 주장한다: student penultimate width ≥ teacher 출력 맵 rank일 때만 성립. 좁으면 residual을 정직히 보고.
- shape mismatch는 crop/pad/SVD로 흡수, rank mismatch는 Skip(학생 원본 유지)이 안전.

## Anti-patterns

- 데이터셋/데이터 로더/데이터용 disk I/O 도입 — hard constraint 위반(여전히 유효).
- **`ferry.build_vocab_map`을 실모델에 직접 사용** — dense `(V_t,V_s)=(256000,48000)` projection 행렬 ≈49GB. 실모델은 `ferry_aster.SparseVocabMap`(index_select gather, 행렬 미생성) 사용.
- **byte-composition seed에 양측 SVD 재사용** — vocab축(256000)을 rank로 압축해 대부분 행 zero-pad(초기 버그). embed은 hidden축만 right-singular 투영, vocab축은 행 인덱스 보존하는 전용 함수 사용.
- **embed 정렬 Procrustes를 iters≥2로 반복** — 앵커 91%가 비한글 + teacher random-probe prior가 anti-Korean → 과회전으로 한글 붕괴(best_rank 9→762). `--iters` 기본 1 고정.
- **균등 가중 Procrustes로 한글을 끌려 함** — 앵커 한글이 9%라 R이 비한글 기하 지배. `--kr-weight`로 한글 앵커 업웨이트가 정석. 단 `e_fit_whole`(lstsq target)은 가중치 무관(앵커별 독립 회귀) → **Procrustes 단계에서만** 가중해야 함(target까지 건드리면 surgical 아님). sweet spot=50(rank 9→2); 과도하게 키워도 iters≥2 붕괴는 그대로.
- **scaled Procrustes(R=s·orthogonal)** — 균일 scale은 softmax 온도만 바꿔 ranking 불변 + byte-comp norm 균등화 파괴. 직교 R만 사용(row-norm 정확 보존).
- bottleneck width plateau를 "버그"로 보고 억지로 100% 맞추려는 시도 — 수학적으로 불가능, honest limit.
- gradient distill에서 **고정 probe 재사용** — 그 probe에 과적합되어 held-out 미개선. 반드시 매 step 새 probe.
- 주의(과거 anti-pattern, DEC-005로 해제됨): "gradient loop 금지"는 더 이상 유효하지 않다. 단, gradient는 stage 3 `distill`에만 두고 stage 1–2는 결정적 closed-form 유지.

## Lessons Learned

- 목표(same-answer)와 제약(no probing)이 충돌하면, 사용자에게 명시적으로 물어 범위를 재정의(synthetic-probe 허용 vs 데이터셋 금지)하는 것이 옳다 (DEC-003).
- **한계는 종류를 구분해야 한다**: rank 벽(폭으로 제거 가능) / 닫힌 형식 한계(gradient로 제거 가능) / 자기회귀 누적(평탄화만). "해결 불가"로 뭉뚱그리지 말고 어떤 제약을 풀면 닫히는지 분류 (DEC-005). 실측 검증: 폭만 키우면 depth mismatch는 ~0.86 정체, distill은 depth-matched ~0.99.

## Gotchas

- **import alias 함정**: `test_ferry.py`는 `import ferry as clone`. 본문 전부 `clone.*` 사용. `ferry.py` 리네임 시 이 alias 라인도 반드시 갱신.
- `report()`의 `mean_relative_error`는 **Stage-1 weight 드리프트**(원본 student init 대비 L2) 지표일 뿐, 답변 품질이 아니다. Stage-1 error가 커도 정상.
- capacity sweep 수치는 데모 teacher에 종속. 현재 deeper teacher `MLP[32,128,96,64,10]` 기준 4/8/16/48 = .471/.467/.562/1.000이며 `theory.html` §6 막대그래프 + §2 SVD 에너지(`net.1.weight` 96×128: top8/16/32 = 24.3/42.6/68.7%)에 하드코딩됨. 데모 teacher 변경 시 동기화 필요.
- **비선형 지원 = `align_hidden`(stage 2b)**: head-only `align_output`은 `ActMLP`(비선형) held-out 일치 불가(student 비선형 특징이 다른 기저). `align_hidden`은 forward sweep으로 student 각 은닉 linear의 pre-activation을 teacher 대응 pre-activation에 `lstsq` 회귀 후 head 정합 — gradient 없음, probe-only. **깊이 일치 시** held-out ~0.97(relu .646→.977, gelu .717→.971, tanh .803→.971); **student가 더 얕으면** 개선되나 부분만(relu .711→.844, gelu .742→.896, tanh .828→.959). scope: flat MLP 계열(`MLP`/`ActMLP`)만; `TinyLM`은 `_linear_chain`이 `[]` → head-only fallback. `theory.html` §7. 선형 `MLP` 정확 보장은 불변.
- **LLM-like(`TinyLM`)**: 순수 torch 트랜스포머(MHA+MLP+LayerNorm+LM head). stage1이 30개 텐서 전이, stage2가 LM head를 토큰 위치별 정합. `agreement`/`align_output`은 `_flatten_logits`로 leading dim을 펴서 `(n,seq,vocab)` 처리(2D는 no-op). closed-form만 쓰면 자기회귀 token-match 감소(.516→.336). **stage 3 `distill`**(매 step 새 token probe)이 per-token 0.41→0.89, 생성 step1 .52→.88·step8 .34→.61로 곡선 평탄화. 단 horizon 증가分 잔존(평탄화≠소거). `theory.html` §8. 전층 정합은 향후 과제.
- **다른 vocabulary = Stage 0 필요**(`reconcile_vocab`, stage 2만으로 불충분): teacher(V=64)/student(V=48)는 LM-head 너비·id 의미가 비교 불가. `VocabMap` 없이 `align_output`/`distill`은 shape error 또는 무관 토큰 비교. 데모(`_demo_vocab_mismatch`, V 64→48 shared-prefix): per-token .410 → .706(align) → .904(distill). student 전용 토큰(`-1`)은 0 타깃 열(teacher 신호 없음). `vocab_map=None`은 엄격 no-op이라 동일-vocab 동작·테스트 불변. `theory.html` §9.
- **현실적 최악 = 세 축 동시**(`_demo_combined_mismatch`): vocab(72→48)·depth(4→2)·width(80→40) 모두 다르고 맵도 scrambled+partial(`_scrambled_vocab_map`, 40/48 임의 슬롯). 단일 축 데모는 각 축 격리; 이건 중첩. scrambled 맵이라 정합 전 top1 ≈ .016(진짜 다른 두 LM baseline; shared-prefix .41과 대비). 파이프라인 회복: stage0 reconcile → stage2 .53 → distill .86. 회귀 테스트 `test_combined_mismatch_baseline_is_near_zero`(base<.10) + `test_combined_mismatch_pipeline_recovers_all_three_axes`(단조, distill>.8). `theory.html` §9.
- `theory.html` 과거 색상 typo `#1a2votes40` → `#1a2440` 수정 이력 있음. hex 손상 주의. **공학자 판본 재작성 후**: 신규 §0(표기법·문제 정식화) 추가로 시각 번호 **0–11**(section 태그 12개), svg 8, table 7(symtab 1 포함). 수식은 `.math`/`.steps`/`.note` CSS(무의존성, MathJax 없음, 유니코드 ℝΣ⊤𝔼). §5에 rank 정리 명문화. 결과 수치는 전부 기존 demo 값 참조(새 하드코딩 0).
- **실모델 `ferry_qwen.py`**: teacher 실제 `Qwen/Qwen3-0.6B`(596M, vocab 151936, hidden 1024, L28, h16/kv8, head_dim 128, tied, RMSNorm/silu), student `ferry-?B`(같은 `Qwen3ForCausalLM`이나 작은 config; ferry-0.1B=512/1536/L8/h4·kv2=103M). **CPU 전용**(`CUDA_VISIBLE_DEVICES=""` import 전 설정, DEC-007 GPU 금지), **데이터-free**(random 토큰 probe), same-vocab → VocabMap 불필요. `LogitsModel` 어댑터로 HF `CausalLMOutput`→raw logits 변환해 `ferry.agreement`/`distill` 재사용. **bfloat16 함정**: Qwen3 config 기본 bf16 → `.float()` 강제 안 하면 CPU autograd backward "Found dtype Float but expected BFloat16". REAL: top1 0.000→0.195, mse 13.0→4.19, plateau ~0.19~0.21(3중 한계: 17% 파라미터+절반 깊이 / 151936 vocab diffuse MSE / off-distribution probe) = 정직한 PoC ceiling(버그 아님). 테스트 6종 gated(`pytest.importorskip('transformers')`). `AGENTS.md` Gotchas 참조.
- **git 커밋 identity = GitHub noreply**(DEC-016): 이 머신 global `user.email` = `17896027+p4r4d0xb0x@users.noreply.github.com`(개인 이메일 `root@ql.gl` 노출 제거). **filter-branch 이메일 스크럽 함정**: `git filter-branch -- --all`은 백업 ref를 **2개** 만든다 — `refs/original/refs/heads/main` **+ `refs/original/refs/remotes/origin/main`**. `git log --all | grep`이 옛 이메일을 계속 잡으면 백업 ref 잔존이 원인 → `refs/original/*` **전부** 삭제 + `reflog expire --expire=now --all` + `gc --prune=now`로 완전 제거. force-push는 `--force-with-lease=main:<old-sha>` 명시적 lease. filter-branch는 author date 보존(`--amend --reset-author`는 날짜를 now로 리셋 → 다중 커밋엔 비권장). 해시 변경 `770bfec→0ec1c4a / 54fbdfc→d664600 / afcf2ef→1a0cd6d`. **`user.name`은 핸들 유지**(실명 박기 = 프라이버시 역행). force-push 후 옛 커밋은 GitHub GC 전까지 직접 SHA로 일시 잔존 가능.

## Gotchas — Aster 확장 (Gemma→Aster, DEC-008~013)

- **Aster forward 재현 정밀 포인트**(`ferry_aster.py`, Rust slm-model 실측): embed lookup **NO sqrt scaling**(HF Gemma는 `*sqrt(hidden)`, Aster는 안 함), **RMSNorm raw gamma**(`gamma*x*rsqrt(mean(x²)+1e-5)`, HF Gemma2의 `(1+gamma)` 아님), **RoPE interleaved GPT-J**(인접 쌍 `2k,2k+1` 회전, HF rotate_half 아님, `rope_dim=head_dim=96` full), GQA 16/8(`kv_head=h//2`), attn scale `1/sqrt(96)`, attn softcap50/final softcap30, GeGLU **gelu-tanh**(주석엔 SwiGLU/SiLU라 적혔으나 실제는 GELU-tanh, DEC-030), tied embed(`logits=h@embed.t()`), 모든 weight `[out,in]`·bias 없음. 하나라도 틀리면 Rust와 parity 깨짐.
- **parity 검증은 `generate --top-k 0 --temperature 0.0 --repetition-penalty 1.0`(순수 greedy)로만**. `slm-cli chat`은 temp=0에서도 기본 `top-k 40` 등 부가처리라 parity 부적합. PyTorch 재현 vs Rust generate가 두 프롬프트에서 **바이트 단위 일치** 확증.
- **Qwen3와 동일 bfloat16 함정**: Gemma도 `load_gemma_teacher`/`build_student`가 `.float()` 강제. 없으면 CPU autograd backward "Found dtype Float but expected BFloat16".
- **embed=0 collapse**(DEC-009): 순수 weight 전이는 embed(vocab 256000≠48000)을 force_skip(zero)해서 token embedding=0 → forward가 position/norm bias만 반영 → teacher/strategy 무관 글자 단위 동일 word-salad. 의미 있는 추론엔 embed 전이 필수.
- **tokenizer 인코딩 차이가 한글 매칭의 근본 원인**(DEC-010/012): Aster=GPT-2 byte-level BPE(공백 `Ġ`, 한글 raw-byte), Gemma=SentencePiece(공백 `▁`, 한글 literal **이나 대부분 `<0xXX>` byte-fallback**). raw 교집합 12.8% → byte-level 정규화 후 41.4%(19880, 한글 1826). 한글 whole-token은 Gemma에 ~2295개뿐이라 NFC/NFD/NFKC 무용 → byte-composition이 유일 해법. 한글 27345 reachable화.
- **reachability ≠ alignment**(DEC-012/013 핵심): byte-comp은 한글 row를 비영 벡터로 만들어 next-token mass 91% 확보(도달)했으나, seed가 Gemma embed singular basis에 살아 Aster hidden 기하와 미정렬 → mass가 diffuse(top1 못 됨). b′ closed-form 직교 Procrustes 정렬이 de-diffuse(best_rank 95→9, top-k40 생존 1/10→8/10)하고 Korean-weighted(`--kr-weight 50`)가 median rank 9→2까지 추가 개선(첫 greedy-Korean 등장)하나 **byte-order 손실+문법 신호 부재는 못 고침**(유창성=학습 영역). greedy top1은 대부분 여전히 비한글(`<pad>`). **Rust에서 한글 보려면**: 정렬 전 `--top-k 0` 필수, 정렬 후 기본 `--top-k 40`에서 생존.
- **`build_vocab_map`은 KD logits SSOT라 미변경 유지**: `ferry_aster.py`의 KD 경로(SparseVocabMap)와 DEC-010/011 재현성이 의존. byte-composition은 그 위에 additive(embed seeding 전용), KD logits에는 무관(logit은 whole-token 단위).
- **big teacher가 순수 전이엔 불리**(DEC-009): FFN SVD energy 2B(hidden2304→1536)=87.8% vs 9B(3584→1536)=70.2%. 축소율이 지배, layer-select(uniform/front) 차이는 0.3%p 미미. tied-embed라 weight-tying이 logits 지배 → greedy 출력에 layer-select 안 드러남.

## Environment Notes

- torch 2.10.0+cu128, Python 3.12.11 사전 설치. toy core는 설치 단계 없음.
- **Aster 확장 대상**: `../SLM_FROM_BEGIN` (Rust from-scratch ko-leading SLM). aster-1b=d1536/L26/h16·kv8/head_dim96/ffn6144/vocab48000, tied embed, safetensors 236 텐서 `v2.*` namespace. live 체크포인트 `artifacts/checkpoints/aster-1b/params.safetensors` (~step3950+ 학습 진행 중, **미접촉**). tokenizer `/data/0A_DATASET/L0_LLM/V3/TOKENIZER/tokenizer.json` (vocab48000).
- **teacher 캐시**: `google/gemma-2-2b`(~5GB, hidden2304/L26/kv4) KD/정렬에 사용, `google/gemma-2-9b`(~37GB, hidden3584/L42/kv8) 전이 실험만. HF 토큰 보유. `_resolve_snapshot(repo)`로 HF 캐시 스냅샷 경로 해석.
- **재생 명령**: 전이 `python transfer_gemma_to_aster.py --teacher google/gemma-2-2b --embed-vocab-map --embed-byte-compose --out ./test_output`; KD `python ferry_aster.py --steps 100 --batch 8 --seq 16`; 정렬 `python align_aster_embed.py --n-batches 16 --iters 1`. chat 실증 `printf '<prompt>\n/exit\n' | ./target/release/slm-cli chat --model <dir> --tokenizer <tok> --model-config configs/model/pretrain-1b.toml --device cpu --max-new-tokens 40 --temperature 1.0 --top-k 0` (../SLM_FROM_BEGIN에서, 한글 보려면 `--top-k 0`).
- 실행: `python -m pytest test_ferry.py -q` (42 cases), `python ferry.py` (toy demo 6-part).
- 실모델: `transformers` 5.10.2 + `accelerate` 1.13.0 **설치됨**, `Qwen/Qwen3-0.6B` HF 캐시에 다운로드됨(~1.2GB). 실행 `python ferry_qwen.py`, `python -m pytest test_ferry_qwen.py -q`. **GPU 금지**(DEC-007, CPU·float32).
- git 저장소 아님. packaging 파일(`pyproject.toml` 등) 없음 — 의도된 단순성(`ferry_qwen.py`만 argparse CLI 보유).
