# Source this file from a strict-mode Bash wrapper. A runner is signalled as a
# process group only after ps proves that the group differs from the wrapper.
RUNNER_PID=${RUNNER_PID:-}
RUNNER_PGID=${RUNNER_PGID:-}
RUNNER_WAIT_STATUS=${RUNNER_WAIT_STATUS:-}
SCRIPT_PGID=${SCRIPT_PGID:-}

runner_initialize() {
  local process_group
  if ! process_group=$(_runner_read_pgid "$$"); then
    echo "Could not identify the wrapper process group." >&2
    return 2
  fi
  SCRIPT_PGID=$process_group
}

runner_install_traps() {
  trap runner_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

runner_cleanup() {
  local exit_status=$?
  trap - EXIT
  if ! runner_stop; then
    echo "Could not fully stop the bounded runner; preserving its temporary files." >&2
  fi
  if [[ -z "$RUNNER_PID" && -z "$RUNNER_PGID" ]]; then
    if [[ -n "${JOB_TMP_ROOT:-}" ]]; then
      rm -rf -- "$JOB_TMP_ROOT"
    fi
  fi
  exit "$exit_status"
}

run_bounded() {
  if (($# == 0)); then
    echo "run_bounded requires a command." >&2
    return 2
  fi
  if [[ -z "$SCRIPT_PGID" ]] && ! runner_initialize; then
    return 2
  fi

  RUNNER_WAIT_STATUS=
  RUNNER_PGID=
  setsid --wait "$@" &
  RUNNER_PID=$!
  if ! _runner_wait_for_owned_group "$RUNNER_PID" 50; then
    if ! _runner_pid_is_running "$RUNNER_PID"; then
      _runner_reap_finished_runner || return 1
      return "$RUNNER_WAIT_STATUS"
    fi
    echo "Could not isolate a process group for the cache and time limits." >&2
    runner_stop || true
    return 2
  fi

  local run_status=0 wait_status=0 cache_bytes elapsed_seconds
  while _runner_pid_is_running "$RUNNER_PID"; do
    if ! cache_bytes=$(du -sb "$JOB_TMP_ROOT" | awk '{print $1}'); then
      echo "Could not measure job-local temporary storage; stopping." >&2
      run_status=1
      break
    fi
    elapsed_seconds=$(( $(date +%s) - JOB_START ))
    if ((cache_bytes > MAX_CACHE_BYTES)); then
      echo "Job-local model/runtime cache exceeded 20 GiB; stopping." >&2
      run_status=1
      break
    fi
    if ((elapsed_seconds >= JOB_BUDGET_SECONDS)); then
      echo "Run reached its 55-minute budget; stopping before the one-hour allocation ends." >&2
      run_status=124
      break
    fi
    sleep 2
  done

  if ((run_status != 0)); then
    runner_stop || echo "Bounded runner teardown did not complete before its deadline." >&2
    return "$run_status"
  fi

  if _runner_pid_is_running "$RUNNER_PID"; then
    echo "Bounded runner did not exit cleanly." >&2
    runner_stop || true
    return 1
  fi
  _runner_reap_finished_runner || return 1
  wait_status=$RUNNER_WAIT_STATUS

  # A successful command must not leave helpers behind in its isolated group.
  if [[ -n "$RUNNER_PGID" ]] && _runner_group_is_running "$RUNNER_PGID"; then
    if ! runner_stop; then
      echo "Bounded command left processes running after teardown." >&2
      return 1
    fi
  else
    RUNNER_PGID=
  fi
  return "$wait_status"
}

runner_stop() {
  local original_pid=${RUNNER_PID:-} stop_status=0
  if [[ -z "$original_pid" && -z "${RUNNER_PGID:-}" ]]; then
    return 0
  fi

  if [[ -n "$original_pid" && -z "${RUNNER_PGID:-}" ]]; then
    _runner_wait_for_owned_group "$original_pid" 50 || true
  fi

  if _runner_group_is_owned; then
    if _runner_group_is_running "$RUNNER_PGID"; then
      _runner_signal_group TERM || true
      if ! _runner_wait_for_group_exit "$RUNNER_PGID" 5; then
        _runner_signal_group KILL || true
        if ! _runner_wait_for_group_exit "$RUNNER_PGID" 5; then
          stop_status=1
        fi
      fi
    fi
  elif [[ -n "$original_pid" ]] && _runner_pid_is_running "$original_pid"; then
    # If isolation cannot be verified, only signal the direct child PID.
    kill -TERM "$original_pid" 2>/dev/null || true
    if ! _runner_wait_for_pid_exit "$original_pid" 5; then
      kill -KILL "$original_pid" 2>/dev/null || true
      if ! _runner_wait_for_pid_exit "$original_pid" 5; then
        stop_status=1
      fi
    fi
  fi

  if [[ -n "$RUNNER_PID" ]] && ! _runner_pid_is_running "$RUNNER_PID"; then
    _runner_reap_finished_runner || stop_status=1
  elif [[ -n "$RUNNER_PID" ]]; then
    stop_status=1
  fi

  if _runner_group_is_owned && _runner_group_is_running "$RUNNER_PGID"; then
    stop_status=1
  elif [[ -n "$RUNNER_PGID" ]]; then
    RUNNER_PGID=
  fi
  return "$stop_status"
}

_runner_read_pgid() {
  local pid=$1 process_group
  if ! process_group=$(ps -o pgid= -p "$pid" 2>/dev/null); then
    return 1
  fi
  process_group=${process_group//[[:space:]]/}
  [[ "$process_group" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$process_group"
}

_runner_pid_is_running() {
  local pid=$1 stat
  if stat=$(ps -o stat= -p "$pid" 2>/dev/null); then
    stat=${stat//[[:space:]]/}
    [[ -n "$stat" && "$stat" != Z* && "$stat" != X* ]]
  else
    kill -0 "$pid" 2>/dev/null
  fi
}

_runner_group_is_running() {
  local pgid=$1 states stat
  if states=$(ps -o stat= -g "$pgid" 2>/dev/null); then
    while IFS= read -r stat; do
      stat=${stat//[[:space:]]/}
      if [[ -n "$stat" && "$stat" != Z* && "$stat" != X* ]]; then
        return 0
      fi
    done <<< "$states"
    return 1
  fi
  kill -0 -- "-$pgid" 2>/dev/null
}

_runner_group_is_owned() {
  [[ "${RUNNER_PGID:-}" =~ ^[0-9]+$ && -n "${SCRIPT_PGID:-}" && "$RUNNER_PGID" != "$SCRIPT_PGID" ]]
}

_runner_signal_group() {
  local signal_name=$1
  _runner_group_is_owned || return 2
  kill "-$signal_name" -- "-$RUNNER_PGID" 2>/dev/null
}

_runner_wait_for_owned_group() {
  local pid=$1 attempts=$2 process_group
  for ((attempt = 0; attempt < attempts; attempt++)); do
    if process_group=$(_runner_read_pgid "$pid") && [[ "$process_group" != "$SCRIPT_PGID" ]]; then
      RUNNER_PGID=$process_group
      return 0
    fi
    if ! _runner_pid_is_running "$pid"; then
      return 1
    fi
    sleep 0.1
  done
  return 1
}

_runner_wait_for_pid_exit() {
  local pid=$1 seconds=$2 deadline
  deadline=$(( $(date +%s) + seconds ))
  while _runner_pid_is_running "$pid"; do
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 0.1
  done
  return 0
}

_runner_wait_for_group_exit() {
  local pgid=$1 seconds=$2 deadline
  deadline=$(( $(date +%s) + seconds ))
  while _runner_group_is_running "$pgid"; do
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 0.1
  done
  return 0
}

_runner_reap_finished_runner() {
  if [[ -z "${RUNNER_PID:-}" ]]; then
    return 0
  fi
  if _runner_pid_is_running "$RUNNER_PID"; then
    return 1
  fi
  if wait "$RUNNER_PID"; then
    RUNNER_WAIT_STATUS=0
  else
    RUNNER_WAIT_STATUS=$?
  fi
  RUNNER_PID=
}
