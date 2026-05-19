#!/usr/bin/env bash
# Wrapper for overnight drift-study runs.
#
# Usage:
#   ./scripts/run_overnight.sh [pairs] [tickets] [interval_seconds] [budget_usd]
#
# Defaults: 3 pairs, 20 tickets per session, 600s probe interval, $20 per-arm budget.
#
# What it does:
#   - activates the venv
#   - uses `caffeinate -i` to prevent idle sleep
#   - uses `nohup` so the process survives terminal close
#   - redirects all output to a log file
#   - prints the PID so you can monitor or kill it
#
# IMPORTANT: closing the laptop LID still sleeps the Mac on most models
# unless you have an external display attached or have explicitly disabled
# clamshell sleep. Either leave the lid open, OR run:
#   sudo pmset -a disablesleep 1
# before this script and:
#   sudo pmset -a disablesleep 0
# when finished.

set -euo pipefail

PAIRS=${1:-3}
TICKETS=${2:-20}
INTERVAL=${3:-600}
BUDGET=${4:-20}
TICKET_SOURCE=${5:-templated}

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv directory not found in $PROJECT_ROOT. Create it with:"
    echo "    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

LOG_DIR="experiments/drift_study/runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight-$(date +%Y%m%d-%H%M%S).log"

echo "Starting overnight drift-study run."
echo "  pairs:            $PAIRS"
echo "  tickets/session:  $TICKETS"
echo "  probe interval:   ${INTERVAL}s"
echo "  per-arm budget:   \$$BUDGET"
echo "  ticket source:    $TICKET_SOURCE"
echo "  log file:         $LOG_FILE"
echo

nohup caffeinate -i python -m experiments.drift_study.run \
    --pairs "$PAIRS" \
    --tickets-per-session "$TICKETS" \
    --probe-interval-seconds "$INTERVAL" \
    --max-budget-usd-per-arm "$BUDGET" \
    --ticket-source "$TICKET_SOURCE" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Background PID: $PID"
echo "Tail with:   tail -f $LOG_FILE"
echo "Kill with:   kill $PID"
echo
echo "caffeinate is preventing IDLE sleep while this PID is alive."
echo "If you close the LID, the Mac may still sleep (see header in this script)."
