#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./ping_litellm_budget.sh --base-url https://your-litellm-host --api-key sk-...

Optional:
  --target-key sk-...
    Query a specific key via ?key=... . This usually requires an admin key in --api-key.
    If omitted, LiteLLM returns info for the auth key itself.

Examples:
  ./ping_litellm_budget.sh \
    --base-url https://api.ai.it.cornell.edu \
    --api-key sk-xxxxxxxx

  ./ping_litellm_budget.sh \
    --base-url https://api.ai.it.cornell.edu \
    --api-key sk-admin-xxxxxxxx \
    --target-key sk-user-xxxxxxxx
EOF
}

BASE_URL=""
API_KEY=""
TARGET_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:-}"
      shift 2
      ;;
    --target-key)
      TARGET_KEY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$API_KEY" ]]; then
  usage >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"
URL="$BASE_URL/key/info"

curl_args=(
  -fsS
  -H "Authorization: Bearer $API_KEY"
  -H "x-litellm-api-key: $API_KEY"
  -H "Accept: application/json"
)

if [[ -n "$TARGET_KEY" ]]; then
  RAW_JSON="$(curl "${curl_args[@]}" -G --data-urlencode "key=$TARGET_KEY" "$URL")"
else
  RAW_JSON="$(curl "${curl_args[@]}" "$URL")"
fi

python3 - <<'PY' "$RAW_JSON"
import json
import sys
from datetime import datetime, timezone

payload = json.loads(sys.argv[1])
info = payload.get("info", {})

spend_raw = info.get("spend")
budget_raw = info.get("max_budget")

spend = float(spend_raw) if spend_raw is not None else 0.0
max_budget = float(budget_raw) if budget_raw is not None else None
remaining = (max_budget - spend) if max_budget is not None else None
used_pct = ((spend / max_budget) * 100.0) if max_budget not in (None, 0.0) else None

now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
print("LiteLLM budget check")
print(f"checked_at_utc: {now_utc}")
print(f"key_alias: {info.get('key_alias')}")
print(f"key_name: {info.get('key_name')}")
print(f"spend: {spend:.2f}")
print(f"max_budget: {max_budget:.2f}" if max_budget is not None else "max_budget: null")
print(f"remaining: {remaining:.2f}" if remaining is not None else "remaining: null")
print(f"used_pct: {used_pct:.2f}%" if used_pct is not None else "used_pct: null")
print(f"budget_reset_at: {info.get('budget_reset_at')}")
print(f"updated_at: {info.get('updated_at')}")
print(f"last_active: {info.get('last_active')}")
PY
