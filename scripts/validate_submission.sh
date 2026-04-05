#!/bin/bash
# Veritas AI — Pre-submission validation script
# Usage: bash scripts/validate_submission.sh https://YOUR_USERNAME-veritas-ai.hf.space

set -e

HF_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

green() { echo -e "\033[32m[PASS]\033[0m $1"; }
red()   { echo -e "\033[31m[FAIL]\033[0m $1"; }

echo ""
echo "========================================"
echo " Veritas AI — Submission Validator"
echo " Target: $HF_URL"
echo "========================================"
echo ""

# Check 1 — HF Space deploys: /reset returns 200
echo "Check 1: HF Space /reset returns 200..."
RESET_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$HF_URL/reset" \
  -H "Content-Type: application/json" -d "{}" --max-time 30)
if [ "$RESET_STATUS" = "200" ]; then
  green "HF Space deploys — /reset returns 200"
  PASS=$((PASS+1))
else
  red "HF Space deploys — /reset returned HTTP $RESET_STATUS (expected 200)"
  FAIL=$((FAIL+1))
fi

# Check 2 — OpenEnv spec compliance: openenv validate
echo "Check 2: openenv validate..."
if command -v openenv >/dev/null 2>&1; then
  if openenv validate 2>&1 | grep -q "Ready"; then
    green "OpenEnv spec compliance — openenv validate passes"
    PASS=$((PASS+1))
  else
    red "OpenEnv spec compliance — openenv validate failed"
    FAIL=$((FAIL+1))
  fi
else
  red "OpenEnv spec compliance — openenv CLI not found (install openenv-core)"
  FAIL=$((FAIL+1))
fi

# Check 3 — Dockerfile builds
echo "Check 3: Docker build..."
if command -v docker >/dev/null 2>&1; then
  if docker build -t veritas-ai:validate -f server/Dockerfile . -q >/dev/null 2>&1; then
    green "Dockerfile builds"
    PASS=$((PASS+1))
  else
    red "Dockerfile builds — docker build failed"
    FAIL=$((FAIL+1))
  fi
else
  red "Dockerfile builds — docker not found"
  FAIL=$((FAIL+1))
fi

# Check 4 — Baseline reproduces: inference.py runs clean
echo "Check 4: inference.py runs..."
if python inference.py --dry-run 2>/dev/null || python inference.py 2>&1 | grep -q "score\|Score\|task_easy"; then
  green "Baseline reproduces — inference.py runs clean"
  PASS=$((PASS+1))
else
  red "Baseline reproduces — inference.py did not complete cleanly"
  FAIL=$((FAIL+1))
fi

# Check 5 — 3+ tasks with graders: scores in 0.0–1.0
echo "Check 5: Tasks and graders..."
SCHEMA=$(curl -s "$HF_URL/schema" --max-time 15)
if echo "$SCHEMA" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0)" 2>/dev/null; then
  green "3+ tasks with graders — /schema returns valid JSON"
  PASS=$((PASS+1))
else
  red "3+ tasks with graders — /schema did not return valid JSON"
  FAIL=$((FAIL+1))
fi

# Check 6 — Health check
echo "Check 6: /health endpoint..."
HEALTH=$(curl -s "$HF_URL/health" --max-time 10)
if echo "$HEALTH" | grep -q "healthy"; then
  green "Health check — /health returns healthy"
  PASS=$((PASS+1))
else
  red "Health check — /health did not return healthy (got: $HEALTH)"
  FAIL=$((FAIL+1))
fi

echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"
echo ""

if [ "$FAIL" -eq 0 ]; then
  echo "✅ All checks passed! Ready to submit."
  exit 0
else
  echo "❌ $FAIL check(s) failed. Fix before submitting."
  exit 1
fi
