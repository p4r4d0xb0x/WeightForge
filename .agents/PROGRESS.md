---
project: Ferry
created: 2026-06-10
updated: 2026-06-12 (public git repo 발행 + strict source-available LICENSE, DEC-015)
---

# Progress

## 2026-06-12 (공개 git repo 발행 — WeightForge, gitignore/README/strict LICENSE, DEC-015)

### Completed

- **빈 공개 repo `p4r4d0xb0x/WeightForge`(PUBLIC, isEmpty) 초기 세팅**(사용자 위임 "잘 세팅해봐").
  로컬 `main` 커밋 0개 → 초기 커밋 `770bfece` → `git push -u origin main`(`[new branch] main -> main`).
- **`.gitignore` 신규**: `test_output/`(**23GB**, 3.8GB `params.safetensors`×6 — GitHub 100MB 초과) +
  모델 가중치(`*.safetensors`/`*.bin`/`*.pt`/...) + 파이썬/테스트 캐시 + venv + 에디터/OS junk 제외.
  실제 푸시 = 소스 5 + 테스트 2 + `theory.html` + `AGENTS.md` + `.agents/` 6 + `README.md` + `LICENSE`
  = **19파일 0.3MB**(>50MB 0개 가드 통과).
- **`README.md` 신규**(최소): 요약·4-stage 표·layout·실행법·정직한 한계·LICENSE 안내. "no README" 규약을
  사용자 결정으로 override(단순성 규칙 유지).
- **`LICENSE` 신규 — Strict Edition**(사용자 verbatim "조항을 더 엄격하게"): source-available·
  publication-reserved. §2 좁은·revocable use(private 비공개·비상업 evaluation/research), §3 금지 9종
  (publication·research-credit·redistribution·commercial·**Work/Output로 타 모델 train/distill(e)**·
  dataset 편입(f)·patent(g)·trademark(h)·competing method(i)), §8 위반 시 파기, §10 injunctive relief,
  §6 아이디어는 저작권으로 못 막음 정직 명시. **OSI 비승인(의도적)**.
- **GitHub 설명문 정정**(`gh repo edit`): 기존 "without ... output matching"(실제와 모순) →
  실제 파이프라인(data-free transfer + closed-form align + gradient distill, depth/width/vocab 변경) 반영.
- **SSOT 동기화**: `AGENTS.md` layout/operating-rules의 stale 문구("no git/README", "Not a git repo")
  갱신, `.agents/DECISION.md` **DEC-015** 추가(MADR-lite, frontmatter count 14→15).

### Notes

- **검증**: `pytest test_ferry.py -q` **42 passed**(코드 미변경), `.gitignore` `git check-ignore`로
  23GB 차단 확인, `git ls-remote` `refs/heads/main = 770bfece`, `gh repo view` `isEmpty:false`·
  `defaultBranch:main`·설명문 반영·PUBLIC.
- **data-free 불변**: 가중치·`test_output` 산출물은 git 진입 불가(gitignore). 모든 `.py`/`theory.html`/
  테스트 로직 **미변경**(코드 변경 0). DEC-004 TODO↔Issue `off` 유지.
- **잔여/heads-up**: 저작권자 표기 = GitHub 핸들 `p4r4d0xb0x`(실명/연락처 후속 교체 가능). 커밋 author
  이메일 `root@ql.gl`이 공개 커밋 메타데이터에 노출(원하면 `--amend --reset-author`로 교체).
- 잔여 마커 0(`todo!()`/`TODO`/skip 없음, 신규/수정 파일 전부).

## 2026-06-12 (Korean-weighted orthogonal Procrustes, `--kr-weight`, DEC-014)

### Completed

- **`align_aster_embed.py` 가중 Procrustes 추가**(DEC-013 surgical 확장): `procrustes_rotation(x, y,
  weights=None)` → `M_cross = (X·w)ᵀ@Y`; `run_align`에서 `anchor_weights = where(kr_mask[whole_ids],
  kr_weight, 1.0)`로 한글 1826개 앵커만 업웨이트. **CLI `--kr-weight`(default 1.0 = DEC-013 완전 재현)**.
  `e_fit_whole`(lstsq target) 불변 — Procrustes 단계에서만 가중. `--iters 1` 고정 유지.
- **Sweep**(`kr_weight ∈ {1,5,20,50} × iters {1,2}`, 10-prompt harness) → **kr_weight=50, iters=1**
  채택. iters≥2는 전반 붕괴(과회전, DEC-013 확인 재현).
- **실행**: `python align_aster_embed.py --kr-weight 50 --iters 1 --n-batches 16 --out
  ./test_output/aster-1b-from-gemma-2-2b-embedmap-bc-aligned-krw50` →
  `{params.safetensors 3.98GB, align_report.json}`. embed-only delta(235 텐서 byte-identical),
  **schema PERFECT MATCH(236, 0/0/0 vs live)**, embed nonzero 47080·|max|1.504(직교 R row-norm 보존).
- **측정(정직, DEC-013 대비 추가 개선)** — survive@40 / median best_kr_rank / greedy-top1-Korean / mass:
  bc `1/10·96·0·0.734` → aligned(DEC-013) `8/10·9·0·0.646` → **krw50(DEC-014) `8/10·2·1·0.721`**.
  median rank **9→2**, 첫 greedy-Korean(`참고`) 1개. 엔드투엔드 top-k40(Rust 기본 디코더 등가): DEC-013
  은 거의 영어(`Hamp islation XNUMX`) vs krw50은 실제 한글 다수(`투명하게/번호/데이트코스/하계/구성해/보증금`).

### Notes

- **정직한 한계(잔여)**: 2/10 프롬프트 회귀 존속(`옛날 옛적에` rank282, `대한민국의 수도는` 4→580),
  greedy 대부분 `<pad>`(한글 rank2 바로 위), 등장 한글 generic·반복(`프랑 딱히 elfare`)·비유창.
  byte-order 손실+문법 신호 부재는 정렬로 불가 = 학습/KD 영역.
- **옵션 랭킹(anti-evasion)**: 1순위 = on-distribution 학습/KD(진짜 천장, **data-free 위반 → 승인 필요,
  미실행 보류**), 2순위 teacher-generated probe KD(data-free, 한글 byte-fallback 벽), 3순위(이번) =
  Korean-weighted Procrustes. FFN hidden-align(C)은 head_dim 256 vs 96 이질성으로 보류.
- 검증: py_compile OK, test_ferry 42/42, 잔여 마커 0. 환경 불변: GPU 금지(DEC-007), data-free,
  live aster-1b ~step3950+ 미접촉(신규 파일만).

## 2026-06-12 (closed-form orthogonal Procrustes embed-basis alignment, b′, DEC-013)

### Completed

- **신규 `align_aster_embed.py`** (data-free, gradient 없이 closed-form). byte-composed embed의
  hidden-basis 미정렬(DEC-012 한계)을 Ferry Stage-2b 정신으로 회전 정렬: whole-token 앵커 19880개를
  synthetic shared-probe로 student `final_hidden` feature `F` 수집 + Gemma-2B teacher logit 앵커 열
  gather → 정규방정식 lstsq `E_fit_whole` → **orthogonal Procrustes `R=UVᵀ`** → `E_aligned=E_seed@R`
  를 전 row 동일 적용(직교 R = row-norm 보존 = byte-comp 균등화 유지).
- **`ferry_aster.py`에 `AsterForCausalLM.final_hidden()` 추가**(forward가 호출 → parity 보존, 검증됨).
- **실행**: `python align_aster_embed.py --n-batches 16 --iters 1` →
  `./test_output/aster-1b-from-gemma-2-2b-embedmap-bc-aligned/{params.safetensors, align_report.json}`.
  embed-only delta(나머지 235 텐서 byte-identical), schema PERFECT MATCH(236, 0/0/0 vs live).
- **측정(정직, partial positive)**: 10개 프롬프트 best-Korean-within-top40(greedy/top-k 생존)
  **BEFORE 1/10 → AFTER 8/10**, 대표 best_kr_rank **95→9**(diffuse 급감), kr_mass 0.91→0.81 유지.
  Rust slm-cli chat **기본 top-k 40**에서 한글 등장(`스톡/웹툰/국민/KBS/삼성/놨`) — bc는 동일 설정 0개.
- **과반복 붕괴 확인**: iters 1→3 best_rank 9→94→762, mass 0.81→0.00 (anti-Korean teacher prior 과회전)
  → `--iters` 기본 1 고정.

### Notes

- **정직한 한계**: 2/10 프롬프트 회귀(`옛날 옛적에` mass→0.002, `대한민국의 수도는` rank 4→732),
  등장 한글이 generic·prompt-비민감(7개 프롬프트 rank9/mass0.807 수렴), greedy top1은 여전히 비한글,
  여전히 word-salad. byte-order 손실+문법 신호 부재는 정렬로 못 고침(학습/KD 영역). b′ = "reachability
  under top-k" 달성, "meaning alignment"는 미달.
- 검증: py_compile OK, test_ferry 42/42(parity refactor 무해), 잔여 마커 0.
- 환경 불변: GPU 금지(DEC-007), data-free, live aster-1b ~step3950+ 미접촉(신규 파일만).

## 2026-06-12 (byte-composition embed seeding으로 한글 커버리지 강화, DEC-012)

### Completed

- **`transfer_gemma_to_aster.py`에 byte-composition seeding 추가** (opt-in `--embed-byte-compose`):
  `_byte_fallback_table` + `build_byte_composition`(미매칭 student 토큰의 UTF-8 바이트를 Gemma
  `<0xXX>` 토큰으로 매핑) + `transfer_embed`에 byte_comp 처리(byte 임베딩 hidden-projected MEAN,
  whole-token row 평균 norm으로 rescale) + `run_transfer`/`print_report`/`main`/`TransferRow`
  (embed_whole/embed_byte) 배선. `build_vocab_map`(KD logits SSOT)은 미변경.
- **진단(결정적)**: 한글 매칭 실패는 정규화 버그가 아니라 Gemma SentencePiece의 byte-fallback 때문.
  Aster whole 한글 토큰 27345 vs Gemma whole 한글 토큰 ~2295. NFC/NFD 무용, NFKC +1 한글.
- **재전이 실행**: `python transfer_gemma_to_aster.py --teacher google/gemma-2-2b --embed-vocab-map
  --embed-byte-compose --out ./test_output` → `./test_output/aster-1b-from-gemma-2-2b-embedmap-bc/`.
  embed 커버리지 **47080/48000 = whole 19880 + byte-composed 27200**(한글 1826→27345),
  FFN energy 87.81% 불변, schema PERFECT MATCH(236, 0/0/0 vs live aster-1b), embed nonzero row-norm
  mean 1.52 std 0.09(rescale 균등화 성공).
- **한글 reachability 실측**: next-token 한글 prob mass **2%→91%**(prompt '옛날 옛적에 한 마을에'),
  full-distribution 샘플링(temp 1.0 ×30) 한글 **0/30→30/30**(실제 단어: 협정/찌개/현장을/요인/게임으로…),
  Rust 런타임 slm-cli `--top-k 0 --temperature 1.0`에서 한글 생성 확인('항공편을 예상했다 성장하고…').

### Notes

- 정직한 결론: 목표(한글 embed 커버리지 상승) **달성**. 단 (1) 91% mass가 27345개 한글 토큰에 diffuse →
  greedy/top-k40은 여전히 structural/영어 선택(**한글 보려면 `--top-k 0` 필수**, 기본 top-k 40은 은폐),
  (2) 한글 토큰 개별 plausible하나 **문법적 coherent 아님**(fluency는 training/KD 필요), (3) byte-mean은
  byte order 손실(crude init). 산 것(한글 reachability)과 한계(diffuse·비유창)를 모두 실측.
- 검증: test_ferry.py 42/42, py_compile OK. live aster-1b ~step3950+ 미접촉(신규 파일만). 잔여 마커 0.

## 2026-06-12 (Aster PyTorch 재현 + 실제 Gemma-2B data-free KD, DEC-011)

### Completed

- **`ferry_aster.py` 추가** (479 LOC, DEC-011): Aster Rust forward를 PyTorch `nn.Module`로 정확 재현.
  Rust 모델은 autograd 불가라 `ferry.distill`(Adam loop)을 쓰려면 PyTorch 재현이 선결 과제.
  data-free(synthetic token probe)·CPU·GPU 금지 유지.
- **재현 요소**: dual-RoPE(interleaved GPT-J, local θ1e4/global θ1e6, full head_dim 96),
  hybrid attn(layer i%5==4 + 마지막 layer global = {4,9,14,19,24,25}, 나머지 sliding 512),
  GQA 16/8, attn soft-cap 50, GeGLU(gelu-tanh), RMSNorm(**raw gamma — Gemma의 (1+γ) 아님**),
  final soft-cap 30, tied embed(**no sqrt scaling — Gemma와 다름**). `SparseVocabMap`으로
  ferry.build_vocab_map의 49GB dense 행렬 회피.
- **parity 바이트 일치 확증** (KD의 전제): `slm-cli generate --top-k 0 --temperature 0.0
  --repetition-penalty 1.0`(순수 greedy)와 두 프롬프트('옛날 옛적에', '대한민국의 수도는 서울이고')에서
  바이트 단위 일치 → ferry_aster.py는 Rust 런타임의 충실한 재현. (chat은 top-k=40 기본값이라 parity 부적합.)
- **KD 실행** (`python ferry_aster.py --steps 100 --batch 8 --seq 16 --lr 1e-3`,
  transfer embedmap student + 실제 Gemma-2B teacher): vocab matched 19880/48000(41.4%, 한글 1826).
  held-out probe agreement **before top1=0.0/mse=109.95/cosine=-0.08 → after top1=0.0/mse=10.72/cosine=0.77**.
  KD 가중치: `./test_output/aster-1b-kd-gemma-2-2b/params.safetensors`.
- **slm-cli chat before/after 실증** (CPU greedy): before(전이)=`the in ( " of … call march phone
  schedule … register opt theless`(인식가능 영단어 다수) vs after(KD)=`and ( - (1 d el2 A a1 on no e
  , / onl the 2 …`(파편·숫자·구두점). 둘 다 word-salad.

### Notes

- 정직한 결론: **logit-space 방향성은 크게 개선(cosine -0.08→0.77)됐으나 fluency 천장은 못 넘음**
  (top1은 probe 위에서도 0, chat 출력은 오히려 더 degenerate). 근본 원인 = KD가 random token probe
  (off-distribution)에서만 teacher를 모방 → 실프롬프트 greedy 디코딩으로 전이 안 됨. 용량 10% +
  vocab 41% + tied-embed basis 미정렬이 겹침. **사전 분석(Qwen 0.20 천장)을 실제 Aster에서 실측 확증** —
  data-free KD로는 이미 from-scratch 학습 중인 Aster를 못 이긴다. on-distribution 데이터가 진짜 해법이나
  data-free 제약상 금지(별도 승인 필요).

## 2026-06-12 (embed vocab-map 전이 — embed=0 collapse 타파, DEC-010)

### Completed

- **`--embed-vocab-map` 추가** (DEC-010): DEC-009에서 발견한 embed=0 collapse를 깨기 위해
  tokenizer 문자열 매칭으로 embed 부분 전이. 데이터-free 유지(tokenizer.json vocab 테이블만 읽음).
- **byte-level vs SentencePiece 정규화**: Aster(GPT-2 byte-level, `Ġ`/raw-byte 한글) ↔
  Gemma(SP, `▁`/literal 한글) 인코딩 차이로 raw 교집합 12.8%(한글 0). `_byte_level_decoder`로
  Aster 토큰을 실제 UTF-8 디코드 + Gemma `▁`→' ' → **정규화 교집합 41.4%(19880, 한글 1826)**.
- `build_vocab_map`/`transfer_embed` 구현: teacher embed를 **hidden축만** right-singular 투영
  (`A V_n`, H_t→H_s), vocab축은 행 인덱스로 보존(양측 SVD가 256000행을 rank로 압축하는 버그 회피).
  매칭 행만 벡터화 scatter, 미매칭 zero. embed nonzero 19880 rows, |max|=2.99.
- **코드 리뷰 후 3건 수정**: (1) `transfer_embed` 48000 Python 루프 → index_select 벡터화,
  (2) `print_report`/docstring을 embed 모드별 **정직 분기**(seeded 비율 + basis-misalignment caveat),
  (3) 초기 버그(`_svd_project` 재사용으로 vocab축 붕괴 → nonzero 1661개) → 전용 right-projection으로 수정.
- **embed=0 collapse 타파 실증 (slm-cli chat, greedy)**: plain 2B(embed=0) =
  `pod 일로 듬 정해진 Critical … 엘리트` vs embed-map 2B =
  `the in ( " of to … register opt theless` → **완전히 다른 출력**. 전이 임베딩이 forward를 구동.
- 검증: schema PERFECT MATCH(236, 0/0/0) 유지, coverage 1.0, ferry 42/42, py_compile OK.
  live aster-1b step3700 미접촉(신규 파일 `./test_output/aster-1b-from-gemma-2-2b-embedmap/`).

### Notes

- 정직한 한계: collapse는 깼으나 여전히 word-salad. seeded embed의 hidden축이 embed 자기
  singular basis라 FFN/attn hidden 회전과 미정렬 → weight tying하 logits geometry 불일치 잔존.
  fluent 전이엔 KD/학습 필요(별도 승인). 매칭 토큰 영문 편중(한글 1826/19880)이 영문 우세 출력 원인.

### 9B-embedmap 확장 실증 (uniform/front 2종, 2026-06-12)

- 잔차 점검 중 "9B는 embed-map 미실행" 갭 발견 → uniform/front 둘 다 embed-map 실행.
  `aster-1b-from-gemma-2-9b-{uniform,front}-embedmap` 생성. 둘 다 schema PERFECT MATCH(236,
  0/0/0), embed matched 19880(41.4%, 한글 1826), FFN energy 70.16% 불변(embed-map은 FFN 무관).
- **chat 실증 3가지 관찰**:
  1. **collapse 타파 재확인**: 9B-embedmap 출력(`HP MB IB FI CV Cat SF …`)이 2B-embedmap
     (`the in ( " of to …`)과 **다름** → embed=0(teacher 무관 동일 collapse)과 달리 embed 전이는
     teacher hidden 차원(3584 vs 2304)에 따라 출력이 갈림. 전이 임베딩이 forward를 실제로 구동.
  2. **uniform == front 글자 단위 동일**: weight-tying된 embed가 logits를 지배 → FFN layer-select
     (깊이 선택) 차이가 greedy 출력에 드러나지 않음. DEC-009의 "shrink ratio 지배, layer-select
     미미"를 추론 측면에서 재확인.
  3. 여전히 word-salad(영문 약어 나열) — basis-misalignment 한계 그대로. KD 없이 fluent 불가.
- 결론: 9B-embedmap은 2B와 **동일 결론**(collapse 타파 O, fluent X). teacher 크기는 embed 매칭
  자체엔 무관(같은 19880), 단 hidden 투영 basis만 달라 출력 텍스트가 갈림. "큰 teacher 불리"는
  FFN 에너지(70% vs 88%)에서만 유효.

## 2026-06-12 (Gemma-2-9B 전이 실험 + 양측직교 SVD 교체, DEC-009)

### Completed

- **양측 직교 SVD 투영으로 `_svd_project` 교체** (DEC-009): 기존 구현은 top-k 재구성 후
  좌상단 블록 슬라이스 → U·V를 인덱스로 잘라 직교성 파괴·선행 행/열 편향(진짜 SVD 아님).
  교체본은 `U_m^T A V_n` (Eckart-Young 최적 양측 저차원 제한). rank=min(M,N)로 캡,
  shape는 항상 crop_pad로 dst 정규화(넓은 축 zero-pad). ferry 테스트 42/42 통과
  (수치 회귀 1건은 distill step 800→1500로 해소 — 0.8 bar 낮추지 않음, 도달점이 더 높아
  수렴만 느려진 것을 5-seed로 규명).
- **레이어 선택 매핑 추가**: `select_teacher_layers(n_student,n_teacher,strategy)` —
  `uniform`(stride) / `front`(앞에서 n개). 42→26 불일치 시 student i→teacher idx.
  `--layer-select {uniform,front}` 인자. 출력 디렉토리명에 teacher+strategy 반영.
- **9B 전이 ×2 실행** (uniform, front): `./test_output/aster-1b-from-gemma-2-9b-{uniform,front}/`.
  둘 다 schema PERFECT MATCH(236 tensors, 0 missing/extra/mismatch).

### 정직한 핵심 발견 (negative result, 예측 부합)

- **teacher가 클수록 순수 weight transfer는 불리**: FFN SVD 에너지 보존
  **2B=87.81% vs 9B=70.16%(uniform)/70.47%(front)**. 9B는 hidden 3584→1536,
  ffn 14336→6144로 축소율이 커 SVD가 버리는 스펙트럼 질량이 더 많음. "teacher 크면
  더 좋다"는 직관과 **반대** — 압축 손실이 지배.
- **uniform vs front 차이 미미** (70.16 vs 70.47%) — layer 선택 전략보다 축소율이 지배 변수.
- **embed zero-skip이 추론 출력을 degenerate 고정점으로 붕괴**시킴 (chat 실증):
  2B·9B-uniform·9B-front 세 모델이 greedy에서 **글자 단위로 동일한 word-salad** 출력.
  수치 확인: 세 출력 모두 `v2.embed.weight |max|=0` (tied lm_head도 zero), 반면 FFN은
  실제로 다름(2B ffn_gate L2=35.32 vs 9B=30.99). 즉 **embed가 죽어 있으면 FFN 전이
  품질 차이(87.8% vs 70.2%)가 추론으로 전혀 드러나지 않는다.** embed=0 → 토큰 임베딩 0
  → forward가 위치/norm bias만 반영 → teacher·전략 무관하게 같은 경로로 붕괴.
- **함의**: 순수 weight transfer로 의미 있는 추론을 보려면 embed 전이가 필수.
  현재 vocab 불일치(256000≠48000)로 막혀 있으므로, 다음 단계는 (a) tokenizer 문자열
  기반 실제 vocab 매칭으로 embed 부분 전이, 또는 (b) KD(별도 승인 필요) 중 택일.

## 2026-06-11 (Gemma-2-2B → Aster aster-1b 순수 가중치 초기 전이, DEC-008)

### Completed

- **`transfer_gemma_to_aster.py` 추가** (DEC-008): 별도 Rust 프로젝트 `../SLM_FROM_BEGIN`의
  from-scratch SLM **Aster aster-1b**를 `google/gemma-2-2b`로부터 **순수 weight-space 초기 전이**.
  KD·forward·probe·gradient·dataset 전부 없음 — `ferry.transform_tensor`의 결정론적 선형대수만 사용.
- teacher 실측: gemma-2-2b가 **L26으로 aster-1b와 레이어 수 정확히 일치**(+soft-cap 50/30,
  GeGLU gelu-tanh activation family 일치) → 9B/Gemma-4보다 깔끔한 1:1 매핑. ~5GB 다운로드.
- **이름 매핑 + semantic 정직성 분리**: meaningful(ffn 78텐서, GeGLU 일치) / partial(norm 53,
  CropPad) / meaningless(embed vocab 256000≠48000, attn head_dim 256≠96+RoPE 불일치 105).
  **embed는 force_skip**(zero-init 유지) — 비교 불가 축에 SVD 노이즈 주입 거부.
- **정직 지표 교체**: zero-init 대비 weight drift는 항상 1.0(무정보) → **SVD 스펙트럼 에너지
  보존율**로 교체. 측정: coverage 0.9958, by_kind {Skip:1, CropPad:53, SvdProject:182},
  **FFN 에너지 보존 87.8%**, attention 에너지 97.95%(단 semantic=meaningless로 분리 →
  높은 에너지≠기능적 의미를 수치로 명시).
- **출력 검증**: `./test_output/aster-1b-from-gemma/{params.safetensors(~3.98GB,f32),
  transfer_report.json}`. 236 텐서 이름+shape이 **실제 aster-1b params와 PERFECT MATCH**
  (missing/extra/mismatch 0) → Aster Rust 런타임 로드 가능 형식 실측 확인. embed all-zero 확인.
- **live checkpoint 안전**: aster-1b step3600 체크포인트 4파일 미접촉(신규 파일로만 출력).
- I/O round-trip(f32/bf16) PASS, 매핑 정합성(target 236 = name_map 236, untouched 0) 검증.

### Notes

- 정직한 결론: 작동 모델이 아닌 **초기 skeleton**. KD/추가학습 없이는 유창한 출력 불가.
  attention·embed 무의미, FFN만 부분 의미 — negative-result를 숨기지 않고 수치로 보고하는 것이 목적.

### 추론 실증 (slm-cli chat, 2026-06-12)

- **Aster Rust 런타임 로드 + forward 성공 실측**: `slm-cli chat --model
  ./test_output/aster-1b-from-gemma --tokenizer .../V3/TOKENIZER/tokenizer.json
  --model-config configs/model/pretrain-1b.toml --device cpu` (CPU, greedy temp=0,
  rep-penalty 1.3). 전이 가중치가 schema 호환되어 **에러 없이 로드·추론까지 완주**
  (vocab 48000 디코딩 정상). `scripts/verify_determinism.sh`의 호출 패턴 참고.
- **출력은 예상대로 word salad** (프롬프트 "옛날 옛적에" → 의미 없는 토큰 난열) —
  KD/학습 0인 초기 skeleton의 정직한 증거.
- **baseline 대조**: live aster-1b **step3700**(from-scratch 학습 중, 스냅샷 떠서 비교)도
  같은 설정에서 아직 word salad — 1B 모델 3700스텝은 극초기라 둘 다 유창하지 않음.
  즉 전이본의 비유창성은 "전이 실패"가 아니라 "학습량 0" 때문이며, **런타임 호환성 자체는
  baseline과 동일하게 성립**함을 실증.
- live training(step3700, 18:05~ 진행 중) 미접촉 — params 스냅샷(`/tmp/opencode/aster_snap`)으로만 비교.

## 2026-06-11 (실모델 Qwen 증류 + theory.html 공학자 판본)

### Completed

- **`ferry_qwen.py` 실모델 확장** (DEC-007, 사용자 "실제 Qwen3-0.6B를 증류해 더 작은 ferry-?B로,
  아키텍처 변경해도 동일한지가 중요, **GPU 사용 금지**"): `transformers`/`accelerate` 설치 +
  Qwen3-0.6B(596M) 다운로드. `LogitsModel` 어댑터(HF `CausalLMOutput` → raw logits 텐서)로
  `ferry.transfer`/`ferry.distill` 재사용. PRESETS `ferry-0.1B`(512/1536/L8/h4·kv2, 103M)/
  `-0.12B`/`-0.2B`. **CPU 전용**(`CUDA_VISIBLE_DEVICES=""`), **데이터-free**(random 토큰 probe),
  same-vocab → VocabMap 불필요.
- **dtype 회귀 수정**: Qwen3 config 기본 bfloat16 → CPU autograd backward가
  "Found dtype Float but expected BFloat16". `load_teacher`/`build_student` 양쪽에 `.float()` 강제.
- REAL(ferry-0.1B): student=teacher의 17.3% 파라미터·절반 깊이. transfer 91/91 coverage 1.0
  (by_kind SvdProject 58 / Copy 16 / CropPad 17), per-token top1 **0.000→0.195** (mse 13.0→4.19,
  cosine 0.07→0.48), ~0.34s/step. **plateau ~0.19~0.21** (3중 한계 누적: ~17% 파라미터+절반 깊이,
  151936-wide vocab의 diffuse MSE, off-distribution random probe) — 정직한 PoC ceiling, 버그 아님.
- `test_ferry_qwen.py` gated 테스트 6종(`pytest.importorskip('transformers')`, 모델 없으면 skip):
  no-GPU, 구조 변경 확인, uniform float32 회귀, LogitsModel 텐서 반환, transfer shape-safe/full-coverage,
  distill data-free 개선. **pytest 6 passed (29s)**, 기존 toy 42 unaffected.
- **`theory.html` 공학자 판본 재작성** (사용자 "공학자가 이해할수있도록 수식·스텝별 설명 추가"):
  신규 **§0 표기법·문제 정식화**(기호표 + 목표식 함수동치/기대위험/argmax) + 각 단계 수식·절차 주입 —
  §1 4단계 steps, §2 SVD/Eckart-Young/E(k), §4 4변환 수식, §5 lstsq 정규방정식 + **rank 정리**,
  §7 forward-sweep pre-activation 회귀, §9 VocabMap 선택행렬·remap·project, §10 기대위험 목적식.
  **무의존성**(MathJax 미사용, 스타일드 HTML+유니코드). 구조 검증: section 12/12, svg 8/8,
  table 7/7, 번호 0–11 순차, hex 무손상, HTML well-formed. 결과 수치 변경 0(전부 기호적).

### Blockers

- None.

### Notes

- 실모델은 toy(TinyLM ~0.89)와 달리 plateau가 명확 — "방향성 전이는 작동하나 fluency 주장 아님".
- 새 의존성/다운로드는 `ferry_qwen.py` 한정 예외(사용자 승인). toy core는 dep-light 유지.
- `theory.html` 섹션 수: 기존 11(§1–11) + 신규 §0 = 시각 번호 0–11(=section 태그 12개).

### Next (정찰로 식별된 열린 과제 — 미착수, 우선순위 후보)

- `ferry_advance.py`: 더 작은 student용 **activation-aware 저차원 전이**(teacher 활성 PCA 부분공간 →
  student 투영). 프로토타입 발견: linear 동폭 AA는 exact; 이점은 **raw-init / tanh·centered 표현**에서
  최대, `align_hidden` 이후엔 init 무의미(정직한 scope). DEFERRED.
- **tokenizer 문자열 기반 `t_for_s` 빌더** — 현재 stage-0 기본은 shared-prefix(데모용), 배포는 실제 매칭 맵 필요.
- **트랜스포머 전층/attention 정합** 또는 **permutation 정렬**(Git Re-Basin류) — 자기회귀 잔차 근본 축소.
- **capacity sweep JSON export** 옵션, **non-MLP(attention) 데모**.
- Qwen plateau 완화 실험(더 긴 CPU distill / target·probe 방식 분석) — ceiling 기준 별도 검토.

## 2026-06-11 (현실적 최악 — 세 축 동시 불일치)

### Completed

- **combined-mismatch 데모+회귀 테스트** (사용자 "LM-HEAD·중간계층 Depth·Vocabulary 셋 다
  다른 경우 성능 확인 필요"): `_scrambled_vocab_map`(임의 슬롯·부분 매핑) + `_demo_combined_mismatch`
  (6번째 데모 파트). teacher `TinyLM(V=72,dim=80,L=4)` → student `(V=48,dim=40,L=2)`, scrambled
  40/48 맵.
- REAL: per-token 정합 전 **0.016**(scrambled = 진짜 다른 두 LM baseline, shared-prefix .41과 대비)
  → stage-0+stage-2 **0.526** → +distill **0.863**.
- 테스트 40→**42**: `test_combined_mismatch_baseline_is_near_zero`(base<.10),
  `test_combined_mismatch_pipeline_recovers_all_three_axes`(단조 상승, distill>.8). pytest 42 passed.
- 문서 동기화: `AGENTS.md` six-part·test 42·gotcha. `theory.html` §9에 combined-mismatch
  소절+표 추가(섹션 수 11 유지, table 5→6). `.agents` MEMORY/TODO 동기화.

### Blockers

- None.

### Notes

- scrambled 맵은 shared-prefix보다 어려운 현실적 baseline. 단일 축 데모는 각 축을 격리하지만
  실제는 세 축이 겹친다. 단계적 파이프라인이 ~0.016 → 0.86으로 회복.

### Next

- 새 워크스트림 `ferry_advance.py`: student가 더 작을 때 효율적 전이(activation-aware 저차원).

## 2026-06-11 (Stage 0 어휘 정합)

### Completed

- **Stage 0 `reconcile_vocab` 구현** (DEC-006, 사용자 요청 "병합 전 vocabulary를 student에 맞추는
  과정 추가"): `VocabMap` 데이터클래스(`t_for_s` + `(V_t×V_s)` 선택 행렬 `projection`) +
  `remap_ids`/`project` + `build_vocab_map` + `shared_token_probe`. `agreement`/`align_output`/
  `distill`에 `vocab_map=None` 옵션 추가(None=엄격한 no-op, 기존 동작·테스트 보존).
- 다른 vocab teacher/student **정합 가능**: 데모(V 64→48 shared-prefix) per-token
  0.410 → 0.706(align) → **0.904**(distill).
- `_demo_vocab_mismatch` 5번째 데모 파트 추가. 테스트 32→**40**(VocabMap 8종: 선택 행렬,
  project/remap, identity no-op, shared probe, 다른 vocab align/distill, continuous 거부).
  pytest 40 passed.
- 문서 동기화: `theory.html` §9(vocab/stage0) 신설·distill §10·결론 §11 이동(구조 11/11·svg 8·
  table 5 검증). `AGENTS.md` Stage 0 파이프라인·gotcha·five-part·test count 40.
  `.agents` DEC-006 + MEMORY/TODO 동기화.

### Blockers

- None.

### Notes

- 핵심: LLM의 출력 축 = vocabulary 자체. 다른 tokenizer면 LM-head 너비·id 의미가 비교 불가 →
  병합 전 어휘 정합이 선행되어야 함. 기본 shared-prefix 맵은 데모용, 배포는 tokenizer 문자열
  매칭 맵 필요. student 전용 토큰(`-1`)은 teacher 신호 없음(정직한 잔차).

### Next

- (선택) tokenizer 문자열 기반 `t_for_s` 빌더 — 현재는 shared-prefix 기본값.
- (선택) 트랜스포머 전층 정합/permutation 정렬 — 자기회귀 잔차 근본 축소.

## 2026-06-11

### Completed

- **Stage 3 `distill` 구현** (DEC-005, 사용자가 no-gradient 제약 명시적 해제): warm-started
  Adam loop, 매 step 새 합성 probe(continuous/token), 데이터-free 유지. 입력 모드 `in_dim` XOR
  (`vocab`+`seq`) 검증.
- 한계 ②③(비선형) **닫힘**: depth-matched ActMLP held-out top-1 head-only .70~.83 →
  hidden+head ~.97 → +distill **.996/.992/.996**(relu/gelu/tanh).
- 한계 ④(자기회귀) **대폭 평탄화**: `TinyLM` per-token 0.41→0.68→**0.89**, 생성 token-match
  step1 .52→**.88**, step8 .34→**.61**.
- `_demo_nonlinear_limit`/`_demo_llm_like`에 distill 비교 추가. 테스트 28→**32**(distill 4종:
  입력 모드 검증, 비선형 닫힘, TinyLM 개선, data-free). pytest 32 passed.
- 문서 동기화: `theory.html` §9(distill) 신설·결론 §10 이동·§7/§8 갱신(구조 10/10·svg 8·table 4
  검증). `AGENTS.md` hard-constraints 갱신(gradient allowed, stage 3 한정)·Stage 3 파이프라인·
  gotcha·test count 32. `.agents` DEC-005 + MEMORY 동기화.

### Blockers

- None.

### Notes

- 핵심 통찰: §7·§8 잔차는 **함수의 한계가 아니라 닫힌 형식 방법의 한계**였다. gradient 제약을
  풀자 닫혔다. 단 자기회귀 horizon 증가分은 평탄화일 뿐 소거는 아님(정직한 잔여 한계).
- 일반화의 키는 **매 step 새 합성 probe** 리샘플(고정 probe는 과적합, 실험 확인).

### Next

- (선택) 트랜스포머 전층(attention/LayerNorm) 정합 또는 permutation 정렬 — 자기회귀 잔차의
  근본 축소. 현재는 향후 과제.

## 2026-06-10

### Completed

- `/init-docs`: `.agents/` 6 docs 생성, TODO↔Issue 정책 **off** 확정 후 `AGENTS.md`에 기록.
- Stage 1 (weight transfer) + Stage 2 (closed-form output alignment) 단일 파일 `ferry.py` 구현 완료.
- `test_ferry.py` 15 cases 전부 통과, `python ferry.py` demo 정상 실행.
- `theory.html` 7섹션 self-contained 이론 문서 작성·구조 검증.
- 파일 리네임 `clone.py`→`ferry.py`, `test_clone.py`→`test_ferry.py` (제품명 Ferry 확정, m0029).

### Blockers

- None.

### Notes

- 핵심 결과(증명한 정리): closed-form `align_output`은 student penultimate width가 teacher 출력 맵을 선형 재구성할 수 있을 때(rank condition)에 한해 `student(x)==teacher(x)`를 보장. 충분 시 held-out `top1_agree=1.0`, `mse≈6e-15`.
- capacity sweep (held-out top-1, out_dim=10): width 4=0.355, 8=0.422, 16=0.520, 48=1.000. 좁으면 plateau — 버그 아닌 honest limit.
- `import ferry as clone` alias 유지가 핵심 함정 (DEC-002, MEMORY 참고).
- `mean_relative_error`는 Stage-1 weight 드리프트 지표일 뿐, 답변 품질 아님.

### Next

- (선택) permutation alignment / 다층 정렬 — TODO.md Medium 참조. 현재 추가 작업 없음.
