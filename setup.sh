#!/usr/bin/env bash
# One-command setup for the match predictor.
#
#   ./setup.sh
#
# Creates an isolated Python environment, installs dependencies, downloads
# ~140,000 historical matches, and fits the model and calibration curves.
# Takes roughly 15 minutes, almost all of it in the training step.
#
# Safe to re-run: it skips work that is already done. Pass --retrain to force
# the model to be refitted, or --update-only to refresh results without
# retraining.

set -euo pipefail

cd "$(dirname "$0")"
GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; RED=$'\033[0;31m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
step() { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$OFF"; }
ok()   { printf "%s  ok%s %s\n" "$GREEN" "$OFF" "$1"; }
warn() { printf "%s  !%s  %s\n" "$YELLOW" "$OFF" "$1"; }
die()  { printf "\n%serror:%s %s\n" "$RED" "$OFF" "$1" >&2; exit 1; }

RETRAIN=0; UPDATE_ONLY=0
for a in "$@"; do
  case "$a" in
    --retrain) RETRAIN=1 ;;
    --update-only) UPDATE_ONLY=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $a" ;;
  esac
done

# ---------------------------------------------------------------- python ---
step "Checking Python"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
    if [ -n "$v" ] && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
[ -n "$PY" ] || die "Python 3.10 or newer is required. Install it from https://python.org and re-run."
ok "$($PY --version 2>&1)"

# ------------------------------------------------------------------- venv ---
step "Setting up an isolated environment (.venv)"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv || die "could not create a virtual environment (you may need: $PY -m pip install virtualenv)"
  ok "created .venv"
else
  ok ".venv already exists"
fi
# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then . .venv/bin/activate            # macOS / Linux
elif [ -f .venv/Scripts/activate ]; then . .venv/Scripts/activate  # Windows (Git Bash)
else die "virtual environment looks broken - delete .venv and re-run"; fi

# --------------------------------------------------------------- packages ---
step "Installing packages"
python -m pip install --quiet --upgrade pip >/dev/null 2>&1 || warn "could not upgrade pip, continuing"
python -m pip install --quiet flask numpy pandas scipy scikit-learn requests pyarrow \
  || die "package install failed - check your internet connection"
ok "flask numpy pandas scipy scikit-learn requests pyarrow"

# ------------------------------------------------------------------- data ---
step "Downloading match data (~140,000 matches, a couple of minutes)"
python -m matchpredictor update \
  || die "download failed - check your internet connection, then re-run ./setup.sh"
ok "results and upcoming fixtures cached"

if [ "$UPDATE_ONLY" = "1" ]; then
  printf "\n%sData refreshed. Model left as it was.%s\n" "$BOLD" "$OFF"; exit 0
fi

# ------------------------------------------------------------------ train ---
if [ -f .mpcache/artifacts/model.pkl ] && [ "$RETRAIN" = "0" ]; then
  step "Model already trained"
  warn "skipping training (pass --retrain to refit)"
else
  step "Training the model (about 12 minutes - this is the slow part)"
  echo "  It refits the model season by season to build the calibration curves."
  echo "  That is what makes its confidence numbers mean something, so it is"
  echo "  worth the wait. You only do this once."
  python -m matchpredictor train || die "training failed - see the output above"
  ok "model and calibration saved to .mpcache/artifacts/"
fi

# ------------------------------------------------------------------- done ---
cat <<EOF

$BOLD================================================================$OFF
$BOLD Ready.$OFF

 Every time you want picks, activate the environment first:

     cd $(pwd)
     source .venv/bin/activate

 Then, three steps:

 ${BOLD}1.${OFF} See what it likes
     python -m matchpredictor slip --days 3 --target 2.0 --bankroll 1000 --currency R

 ${BOLD}2.${OFF} Make a price sheet, then fill in Betway's prices
     python -m matchpredictor prices --days 3 --out betway_prices.json

 ${BOLD}3.${OFF} Feed them back for real picks and a staking plan
     python -m matchpredictor slip --days 3 --prices betway_prices.json \\
         --real-prices-only --target 2.0 --bankroll 1000 --currency R

 Refresh results before each match day:
     ./setup.sh --update-only

 Read MATCH_PREDICTOR.md before staking anything. Over 13 backtested
 seasons this lost 1.0% per bet. It ranks outcomes honestly; it has no
 proven ability to beat Betway's price.
$BOLD================================================================$OFF
EOF
