#!/usr/bin/env bash
# One live day of a quantlab strategy on the IBKR paper account (#52).
#
# cron calls this every hour (Debian cron has no CRON_TZ); it checks the time
# in New York itself, so daylight-saving changes need no edit:
#
#   0 * * * * $HOME/projects/quantlab-ibkr/scripts/live_daily.sh
#
# 06 ET  morning: wait until quantlab's daily data update (its own cron,
#        quantlab scripts/data_update/update_daily.sh) has written "done" for today
#        in <DATA_DIR>/update_status.json, checking every RETRY_MINUTES until
#        MORNING_CUTOFF (a day the vendor publishes nothing new is never
#        carried forward); then quantlab's daily prediction job; then, from
#        DECIDE_AT (07:15), `quantlab-ibkr live decide`
#        (market-on-open orders before the 09:20 deadline). Only if the auction
#        is missed (the decide step reaches LATE_AT, or decide is refused for
#        its order deadline) does it decide again from AFTER_OPEN_AT with
#        --after-open: day market orders after the open.
#        The prediction job checks every input it reads holds t.
# 10 ET  record: `quantlab-ibkr live record` (fills, then the Decision recheck).
#
# A step that has not succeeded by its cut-off holds the day: no order is
# sent on stale data. Weekends are skipped; on a market holiday the decide
# step finds t already decided and does nothing.
#
# Environment (files readable by the owner only):
#   ~/.config/quantlab/ibkr.env      TWS_ACCOUNT (the paper account, DU...)
# Overrides: LIVE_DIR, QUANTLAB_DIR, IBKR_DIR, DATA_DIR, CPUS, the times
# below, DECIDE_FLAGS (e.g. "--dry-run": decide and print, submit nothing),
# and MIRROR (the store predict_day.py mirrors from the price store; the S&P 500
# strategy's prices slice by default).
# `live_daily.sh morning|record` runs one step now.
set -u

DATA_DIR=${DATA_DIR:-/data/quantlab}
LIVE_DIR=${LIVE_DIR:-$DATA_DIR/live/sp500_xgb_mvo}
QUANTLAB_DIR=${QUANTLAB_DIR:-$HOME/projects/quantlab2}
IBKR_DIR=${IBKR_DIR:-$HOME/projects/quantlab-ibkr}
CPUS=${CPUS:-64-127}
PY=$IBKR_DIR/.venv/bin/python
CONFIG=$LIVE_DIR/live.json
RETRY_MINUTES=${RETRY_MINUTES:-15}
MORNING_CUTOFF=${MORNING_CUTOFF:-08:30}
# After the IB Gateway's daily restart at 07:00 (~/ib-gateway/restart_daily.sh);
# the contract lookup and the node take ~20 minutes for ~3,000 symbols.
DECIDE_AT=${DECIDE_AT:-07:15}
LATE_AT=${LATE_AT:-09:00}
AFTER_OPEN_AT=${AFTER_OPEN_AT:-09:31}
MIRROR=${MIRROR:-$DATA_DIR/market/sharadar/sp500_prices/sp500_prices.zarr}

ny() { TZ=America/New_York date "$@"; }
now_hm() { ny +%H:%M; }
log_file="$LIVE_DIR/logs/$(ny +%F).log"
mkdir -p "$LIVE_DIR/logs"
log() { echo "$(ny '+%F %T %Z') $*" >> "$log_file"; }
run() { taskset -c "$CPUS" "$@" >> "$log_file" 2>&1; }

load_env() {
    # shellcheck disable=SC1090
    [ -r "$1" ] && { set -a; . "$1"; set +a; }
}

# Run "$@" until it exits 0 (or one of the accepted codes in OK_CODES), retrying
# every RETRY_MINUTES until MORNING_CUTOFF. Returns the last status.
retry_until_cutoff() {
    local name=$1; shift
    local status
    while :; do
        log "$name: start"
        "$@"; status=$?
        case " ${OK_CODES:-0} " in *" $status "*) log "$name: ok ($status)"; return 0 ;; esac
        log "$name: exit $status"
        if [[ "$(now_hm)" > "$MORNING_CUTOFF" ]]; then
            log "$name: past $MORNING_CUTOFF, giving up; the day holds"
            return "$status"
        fi
        sleep $((RETRY_MINUTES * 60))
    done
}

# Whether quantlab's data update has written "done" for today (New York date).
data_ready() {
    "$PY" -c 'import json,sys
s = json.load(open(sys.argv[1]))
sys.exit(0 if s.get("date") == sys.argv[2] and s.get("state") == "done" else 2)' \
        "$DATA_DIR/update_status.json" "$(ny +%F)" 2>/dev/null
}

wait_and_predict() {
    data_ready || { log "data update: not done for $(ny +%F) yet"; return 2; }
    predict_day
}

predict_day() {
    local run_dir
    run_dir=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['quantlab_run'])" "$CONFIG")
    (cd "$QUANTLAB_DIR" &&
     QUANTLAB_DATA_DIR=$DATA_DIR run "$PY" scripts/live/predict_day.py "$run_dir" \
         --store "$LIVE_DIR/live_predictions.zarr" \
         --mirror "$MIRROR" \
         --may-lag "$DATA_DIR/market/fred/fred_dtb3_1d/fred_dtb3_1d.zarr")
}

live() {
    local step=$1; shift
    (load_env "$HOME/.config/quantlab/ibkr.env"
     cd "$IBKR_DIR" && QUANTLAB_DATA_DIR=$DATA_DIR run .venv/bin/quantlab-ibkr live "$step" "$CONFIG" "$@")
}

morning() {
    exec 9> "$LIVE_DIR/.morning.lock"
    flock -n 9 || { log "morning: already running"; return 0; }
    log "morning: begin"
    # predict_day.py: 0 appended, 3 already predicted, 2 data missing (retry).
    OK_CODES="0 3" retry_until_cutoff "data and prediction job" wait_and_predict || return 1
    while [[ "$(now_hm)" < "$DECIDE_AT" ]]; do sleep 60; done
    if [[ ! "$(now_hm)" < "$LATE_AT" ]]; then
        log "decide: $LATE_AT passed, the opening auction is out of reach"
        after_open; return $?
    fi
    log "decide: start ${DECIDE_FLAGS:-}"
    local mark; mark=$(wc -l < "$log_file")
    # shellcheck disable=SC2086
    live decide ${DECIDE_FLAGS:-}; local status=$?
    log "decide: exit $status"
    if [ "$status" -ne 0 ] && tail -n +"$mark" "$log_file" | grep -q "order deadline"; then
        log "decide: refused for its order deadline"
        after_open; return $?
    fi
    return "$status"
}

# The opening auction was missed: decide with day market orders after the open.
after_open() {
    while [[ "$(now_hm)" < "$AFTER_OPEN_AT" ]]; do sleep 30; done
    log "decide --after-open: start ${DECIDE_FLAGS:-}"
    # shellcheck disable=SC2086
    live decide --after-open ${DECIDE_FLAGS:-}; local status=$?
    log "decide --after-open: exit $status"
    return "$status"
}

record() {
    exec 9> "$LIVE_DIR/.record.lock"
    flock -n 9 || { log "record: already running"; return 0; }
    log "record: start"
    live record; local status=$?
    log "record: exit $status"
    return "$status"
}

case "${1:-cron}" in
    morning) morning ;;
    record) record ;;
    cron)
        [ "$(ny +%u)" -le 5 ] || exit 0
        case "$(ny +%H)" in
            06) morning ;;
            10) record ;;
        esac
        ;;
    *) echo "usage: $0 [cron|morning|record]" >&2; exit 2 ;;
esac
