# Bisect Diffusion Performance Regression

Find the exact PR that introduced a performance regression in SGLang diffusion tests by running real GPU benchmarks and binary searching through commit history.

## Slash Command

`/bisect-diffusion-perf <case_id> [step_size]`

## When to Use This Skill

- A diffusion perf test shows E2E latency regression compared to older commits
- You need to find which commit/PR caused the regression on real hardware
- CI perf checks are failing with "Validation failed for 'E2E Latency'"

## Arguments

- **case_id** (required): Test case ID from `testcase_configs.py`, e.g. `flux_2_image_t2i`, `fsdp-inference`, `hunyuan3d_shape_gen`, `flux_2_image_t2i_2_gpus`
- **step_size** (optional): Initial exponential backoff step for finding GOOD commit. Default `50`.

---

## Environment Setup

### Prerequisites

```bash
# 1. Clone and install
git clone https://github.com/sgl-project/sglang.git && cd sglang
pip install -e "python[all]"

# 2. Required env vars
export FLASHINFER_DISABLE_VERSION_CHECK=1
# For gated models (FLUX.2-dev, etc.):
export HF_TOKEN=<your_token>
```

### Ensure GPU is Idle

**Before every benchmark run**, confirm no other processes occupy the GPU:

```bash
nvidia-smi
# Check: all GPU memory should be free, no processes listed under "Processes"
# If something is running, either kill it or use CUDA_VISIBLE_DEVICES to pick a free GPU

# To select specific GPUs (e.g. for 2-GPU tests):
export CUDA_VISIBLE_DEVICES=0,1
```

> **Critical**: GPU contention will corrupt your measurements and make bisecting impossible. Always verify GPU is idle.

### Warm-up and Stability

- The test framework includes warmup by default (first request excluded from timing).
- For extra confidence, you can run the same test **twice** and compare. If results differ by >5%, something is interfering (thermal throttle, background process, etc.).

---

## Test Case Reference

### Which test file for which case?

Look up the case in `python/sglang/multimodal_gen/test/server/testcase_configs.py`:

| List | Test File | GPU Count |
|------|-----------|:---------:|
| `ONE_GPU_CASES_A` | `test_server_a.py` | 1 |
| `ONE_GPU_CASES_B` | `test_server_b.py` | 1 |
| `TWO_GPU_CASES_A` | `test_server_c.py` | 2 |
| `TWO_GPU_CASES_B` | `test_server_d.py` | 2 |

Common cases:

| case_id | List | GPUs | ~Duration | Model |
|---------|------|:----:|:---------:|-------|
| `zimage_image_t2i` | ONE_GPU_CASES_A | 1 | ~1s | Z-Image-Turbo (small) |
| `flux_2_klein_image_t2i` | ONE_GPU_CASES_A | 1 | ~0.4s | FLUX.2-klein-4B |
| `flux_image_t2i` | ONE_GPU_CASES_A | 1 | ~8s | FLUX.1-dev |
| `flux_2_image_t2i` | ONE_GPU_CASES_A | 1 | ~26s | FLUX.2-dev |
| `fsdp-inference` | TWO_GPU_CASES_A | 2 | ~3s | Z-Image-Turbo + FSDP |
| `hunyuan3d_shape_gen` | ONE_GPU_CASES_B | 1 | ~250-330s | Hunyuan3D-2 |
| `flux_2_image_t2i_2_gpus` | TWO_GPU_CASES_B | 2 | ~28s | FLUX.2-dev TP=2 |

### How to Run a Single Case

```bash
# Use SGLANG_GEN_BASELINE=1 to print perf metrics without asserting against baselines.
# This is essential — we don't care about CI baselines, only the raw E2E number.
SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/{test_file}.py -k {case_id} 2>&1 | tee /tmp/perf_{case_id}.log
```

### How to Read the E2E Result

The test outputs a baseline JSON block when `SGLANG_GEN_BASELINE=1`. Look for:

```
"expected_e2e_ms": 30798.39
```

Or search the log:

```bash
grep "expected_e2e_ms" /tmp/perf_{case_id}.log
```

This is the actual measured E2E latency on your hardware for that commit.

### Logging Convention

**Every single run must be recorded.** Use the following one-liner after each test to append to a persistent log file. No run should go unrecorded.

```bash
LOGFILE=/tmp/bisect_${CASE_ID}.csv

# Initialize log (only once at start)
echo "timestamp,case_id,commit_sha,commit_msg,e2e_ms,vs_head_pct,verdict,notes" > $LOGFILE

# After each run, append result (fill in E2E_MS and VERDICT manually or via script)
COMMIT_SHA=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --format="%s" | head -c 80)
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
E2E_MS=<from grep output>
echo "$TIMESTAMP,${CASE_ID},$COMMIT_SHA,\"$COMMIT_MSG\",$E2E_MS,<vs_head_pct>,<GOOD|BAD|NOISE>,\"\"" >> $LOGFILE
```

Or use this helper function (paste into shell at start of session):

```bash
log_run() {
    local CASE_ID=$1 E2E_MS=$2 VERDICT=$3 NOTES="${4:-}"
    local SHA=$(git rev-parse --short HEAD)
    local MSG=$(git log -1 --format="%s" | cut -c1-80)
    local TS=$(date +%Y-%m-%dT%H:%M:%S)
    local VS_HEAD=""
    if [ -n "$HEAD_E2E" ]; then
        VS_HEAD=$(echo "scale=1; ($E2E_MS - $HEAD_E2E) * 100 / $HEAD_E2E" | bc)%
    fi
    printf "%-20s %-10s %-8s %-10s %-8s %-6s %s\n" "$TS" "$CASE_ID" "$SHA" "${E2E_MS}ms" "$VS_HEAD" "$VERDICT" "$NOTES"
    printf "%-20s %-10s %-8s %-10s %-8s %-6s %s\n" "$TS" "$CASE_ID" "$SHA" "${E2E_MS}ms" "$VS_HEAD" "$VERDICT" "$NOTES" >> /tmp/bisect_${CASE_ID}.log
}

# Usage after each test:
# log_run flux_2_image_t2i 30798 BAD "HEAD baseline"
# log_run flux_2_image_t2i 25100 GOOD "step -150"
```

At the end of bisecting, print the full log:

```bash
cat /tmp/bisect_${CASE_ID}.log
```

---

## Workflow

### Phase 1: Measure Current HEAD

**Goal**: Record the E2E latency on HEAD as the BAD reference.

```bash
CASE_ID={case_id}   # set once for the session
TEST_FILE={test_file}

cd /path/to/sglang
git checkout main && git pull
nvidia-smi  # verify GPU idle

SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/${TEST_FILE}.py -k $CASE_ID 2>&1 | tee /tmp/perf_head.log
grep "expected_e2e_ms" /tmp/perf_head.log

# Record HEAD as the BAD baseline for all future comparisons
HEAD_E2E=<value from grep>
HEAD_SHA=$(git rev-parse --short HEAD)
log_run $CASE_ID $HEAD_E2E BAD "HEAD baseline"
```

### Phase 2: Find a GOOD Commit (Exponential Backoff)

**Goal**: Walk backwards from HEAD until you find a commit where E2E is **significantly lower** than HEAD.

**Definition of GOOD**: E2E is at least 10% lower than HEAD_E2E. This threshold accounts for natural run-to-run variance while catching real regressions.

```
GOOD if: actual_e2e < HEAD_E2E * 0.90
BAD if:  actual_e2e >= HEAD_E2E * 0.95
NOISE if: in between (re-run to confirm)
```

```bash
STEP=50
while true; do
    SHA=$(git log --oneline -1 --skip=$STEP --format="%H" main)
    echo "=== Testing $STEP commits back: $(git log --oneline -1 $SHA) ==="

    git checkout $SHA
    pip install -e "python[all]" --no-build-isolation 2>/dev/null
    nvidia-smi  # verify GPU idle

    SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/${TEST_FILE}.py -k $CASE_ID 2>&1 | tee /tmp/perf_step${STEP}.log
    E2E=$(grep "expected_e2e_ms" /tmp/perf_step${STEP}.log | grep -oP '[\d.]+')
    echo "E2E: ${E2E}ms"

    # Log this run immediately
    log_run $CASE_ID $E2E "<GOOD|BAD>" "backoff step -${STEP}"

    # Check verdict: GOOD if <90% of HEAD, BAD if >=95%
    # Manually compare: if E2E < HEAD_E2E * 0.90 → break (GOOD found)
    # Otherwise: increase step and continue
    STEP=$((STEP * 2 + 50))
done
```

> **Tip**: If the regression is very recent (known from CI within last week), start with STEP=20. If unknown age, start with STEP=50.

### Phase 3: Binary Search

**Goal**: Narrow down between GOOD_SHA and BAD_SHA to 1-3 commits.

```bash
GOOD_SHA=<the GOOD commit from Phase 2>
BAD_SHA=<the nearest BAD commit from Phase 2, or HEAD>

# How many commits in range?
RANGE=$(git log --oneline $GOOD_SHA..$BAD_SHA | wc -l)
echo "Range: $RANGE commits"
```

Each iteration, test the midpoint:

```bash
ITER=1
while true; do
    RANGE=$(git log --oneline $GOOD_SHA..$BAD_SHA | wc -l)
    echo "=== Bisect iter $ITER | Range: $RANGE commits | GOOD=$GOOD_SHA BAD=$BAD_SHA ==="
    [ "$RANGE" -le 3 ] && echo "Range small enough, go to Phase 4" && break

    MID=$(git log --oneline --format="%H" $GOOD_SHA..$BAD_SHA | awk "NR==int(($RANGE+1)/2)")
    echo "Midpoint: $(git log --oneline -1 $MID)"

    git checkout $MID
    pip install -e "python[all]" --no-build-isolation 2>/dev/null
    nvidia-smi  # verify GPU idle

    SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/${TEST_FILE}.py -k $CASE_ID 2>&1 | tee /tmp/perf_bisect_${ITER}.log
    E2E=$(grep "expected_e2e_ms" /tmp/perf_bisect_${ITER}.log | grep -oP '[\d.]+')
    echo "E2E: ${E2E}ms"

    # Log this run
    log_run $CASE_ID $E2E "<GOOD|BAD>" "bisect iter $ITER, range=$RANGE"

    # Update boundaries based on verdict:
    # If BAD → BAD_SHA=$MID
    # If GOOD → GOOD_SHA=$MID
    ITER=$((ITER + 1))
done
```

### Phase 4: Confirm the PR

When range is 1-3 commits:

```bash
git log --oneline $GOOD_SHA..$BAD_SHA
```

For each suspect, do a definitive before/after pair:

```bash
SUSPECT=<sha from git log output>
echo "=== Confirming suspect: $(git log --oneline -1 $SUSPECT) ==="

# BEFORE (parent of suspect)
git checkout ${SUSPECT}~1
pip install -e "python[all]" --no-build-isolation 2>/dev/null
nvidia-smi
SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/${TEST_FILE}.py -k $CASE_ID 2>&1 | tee /tmp/perf_before.log
BEFORE_E2E=$(grep "expected_e2e_ms" /tmp/perf_before.log | grep -oP '[\d.]+')
log_run $CASE_ID $BEFORE_E2E "<GOOD|BAD>" "confirm: ${SUSPECT}~1 (before)"

# AFTER (suspect itself)
git checkout ${SUSPECT}
pip install -e "python[all]" --no-build-isolation 2>/dev/null
nvidia-smi
SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/${TEST_FILE}.py -k $CASE_ID 2>&1 | tee /tmp/perf_after.log
AFTER_E2E=$(grep "expected_e2e_ms" /tmp/perf_after.log | grep -oP '[\d.]+')
log_run $CASE_ID $AFTER_E2E "<GOOD|BAD>" "confirm: ${SUSPECT} (after)"

# Summary
echo "================================================"
echo "SUSPECT: $(git log -1 --format='%h %s' $SUSPECT)"
echo "BEFORE:  ${BEFORE_E2E}ms"
echo "AFTER:   ${AFTER_E2E}ms"
echo "DELTA:   $(echo "scale=1; ($AFTER_E2E - $BEFORE_E2E) * 100 / $BEFORE_E2E" | bc)%"
echo "================================================"
```

If BEFORE is GOOD and AFTER is BAD → **this is the regression commit**.

Extract PR number from commit message:
```bash
git log -1 --format="%s" $SUSPECT
# Output like: "[diffusion] some change (#12345)"
```

### Phase 5: Print Full Log and Move to Next Case

```bash
echo ""
echo "========== BISECT COMPLETE: $CASE_ID =========="
echo ""
cat /tmp/bisect_${CASE_ID}.log
echo ""
echo "RESULT: $(git log -1 --format='%h %s' $SUSPECT)"
echo "BEFORE: ${BEFORE_E2E}ms → AFTER: ${AFTER_E2E}ms"
echo "================================================"
```

Save the full log for later reference. Then set new `CASE_ID` and `TEST_FILE` and return to Phase 1 for the next case.

> **Shortcut**: If multiple cases are suspected to share the same root cause (e.g. all are image models, or all regressed around the same date), test the second case ONLY at the confirmed regression commit's before/after — no need to fully re-bisect. Just run Phase 4 directly on the already-identified suspect.

---

## Troubleshooting

### Test fails to start / import error after checkout

Old commits may have different dependencies. Try:

```bash
pip install -e "python[all]" --no-build-isolation
# If that fails:
pip install -e "python[diffusion]"
```

### Model download is slow

Models are cached in `~/.cache/huggingface/hub/`. Once downloaded for the first test, subsequent runs at different commits reuse the cache. Pick a small-model case for the first bisect (`fsdp-inference` or `zimage_image_t2i`).

### E2E varies >5% between identical runs

Something is wrong with the environment:
1. Check `nvidia-smi` — another process may have started
2. Check GPU temperature: `nvidia-smi -q -d TEMPERATURE` — throttling above ~83°C
3. Check power limit: `nvidia-smi -q -d POWER`
4. Wait for GPU to cool down between runs
5. If variance persists, run 3 times and take the median

### `pip install` takes too long

If the commit only changes `.py` files (no C++/CUDA), editable install is instant. If it touches `sgl-kernel` or C extensions, the build is unavoidable. You can skip reinstall for pure-Python changes:

```bash
# Check what the commit changed
git diff --name-only {SHA}~1 {SHA}
# If only .py files → skip pip install
```

### 2-GPU case: which GPUs?

```bash
export CUDA_VISIBLE_DEVICES=0,1
# Verify both are idle:
nvidia-smi -i 0,1
```

---

## Output Template

```markdown
## Diffusion Performance Bisect Report

### Environment
- GPU: {gpu_model} x{count}
- Driver: {nvidia-smi driver version}
- CUDA: {cuda version}
- Python: {python --version}
- Date: {date}

### Results

| Case | Regression PR | Before (ms) | After (ms) | Delta |
|------|--------------|------------:|------------:|------:|
| {case_1} | #{PR} `{title}` | {before} | {after} | +{pct}% |
| {case_2} | #{PR} `{title}` | {before} | {after} | +{pct}% |

### Full Run Log: {case_id}
(paste from /tmp/bisect_{case_id}.log)

TIMESTAMP            CASE_ID    COMMIT   E2E        vs HEAD  VERDICT NOTES
2026-03-26T20:01:00  flux_2..   abc1234  30798ms    —        BAD     HEAD baseline
2026-03-26T20:15:00  flux_2..   def5678  30650ms    -0.5%    BAD     backoff step -50
2026-03-26T20:30:00  flux_2..   9ab0123  25100ms    -18.5%   GOOD    backoff step -150
2026-03-26T20:45:00  flux_2..   bcd2345  28900ms    -6.2%    BAD     bisect iter 1, range=100
2026-03-26T21:00:00  flux_2..   efg6789  25300ms    -17.9%   GOOD    bisect iter 2, range=50
...
2026-03-26T21:30:00  flux_2..   hij3456  25200ms    -18.2%   GOOD    confirm: xyz7890~1 (before)
2026-03-26T21:45:00  flux_2..   xyz7890  30100ms    -2.3%    BAD     confirm: xyz7890 (after)
```
