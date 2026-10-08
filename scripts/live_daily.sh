#!/usr/bin/env bash
# One live day of a quantlab strategy on the IBKR paper account (#52).
#
# cron calls this every hour (Debian cron has no CRON_TZ); it checks the time
# in New York itself, so daylight-saving changes need no edit:
#
#   0 * * * * $HOME/projects/quantlab-ibkr/scripts/live_daily.sh
#
# 06 ET  morning: Sharadar update.py (--rebuild-dropped), then quantlab's
#        daily prediction job; the pair is retried every RETRY_MINUTES until
#        the job has t's row or MORNING_CUTOFF passes (a late vendor table,
#        such as SP500 membership a day behind SEP, is waited for, never
#        carried forward); then, from DECIDE_AT, `quantlab-ibkr live decide`.
#        update.py's own exit status does not stop the morning: a store the
#        strategy does not read may fail, and the job checks every input it
#        reads holds t.
# 10 ET  record: `quantlab-ibkr live record` (fills, then the Decision recheck).
#
# A step that has not succeeded by its cut-off holds the day: no order is
# sent on stale data. Weekends are skipped; on a market holiday the decide
# step finds t already decided and does nothing.
#
# Environment (files readable by the owner only):
#   ~/.config/quantlab/sharadar.env  SHARADAR_API_KEY (read by update.py only)
#   ~/.config/quantlab/ibkr.env      TWS_ACCOUNT (the paper account, DU...)
# Overrides: LIVE_DIR, QUANTLAB_DIR, IBKR_DIR, DATA_DIR, CPUS, the times
# below, and DECIDE_FLAGS (e.g. "--dry-run": decide and print, submit nothing).
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
DECIDE_AT=${DECIDE_AT:-08:35}

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

update_sharadar() {
    (load_env "$HOME/.config/quantlab/sharadar.env"
     cd "$QUANTLAB_DIR" &&
     QUANTLAB_DATA_DIR=$DATA_DIR run "$PY" scripts/sharadar/update.py --rebuild-dropped \
         --download-dir "$DATA_DIR/downloads" --zarr-dir "$DATA_DIR/zarrs")
}

# One round: update the vendor stores (status logged, not used), then the
# prediction job, whose status is the round's.
update_and_predict() {
    update_sharadar; log "sharadar update: exit $?"
    predict_day
}

predict_day() {
    local run_dir
    run_dir=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['quantlab_run'])" "$CONFIG")
    (cd "$QUANTLAB_DIR" &&
     QUANTLAB_DATA_DIR=$DATA_DIR run "$PY" scripts/live/predict_day.py "$run_dir" \
         --store "$LIVE_DIR/live_predictions.zarr" \
         --mirror "$DATA_DIR/pipeline/sharadar_sp500/prices.zarr" \
         --may-lag "$DATA_DIR/zarrs/fred_dtb3_1d.zarr")
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
    OK_CODES="0 3" retry_until_cutoff "update and prediction job" update_and_predict || return 1
    while [[ "$(now_hm)" < "$DECIDE_AT" ]]; do sleep 60; done
    log "decide: start ${DECIDE_FLAGS:-}"
    # shellcheck disable=SC2086
    live decide ${DECIDE_FLAGS:-}; local status=$?
    log "decide: exit $status"
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
