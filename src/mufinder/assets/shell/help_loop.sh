## Prints every keyword description between marker lines.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. A placeholder alone on a line that is given no text
## removes the line. Lines starting ## are removed.
##
## tpl_query_timeout       default seconds one query may take
## tpl_openmpi_settings    NAME=value pairs that keep Open MPI in-process
## tpl_castep              the CASTEP invocation, launcher included
## tpl_timing              help_loop_timing.sh, or nothing
## tpl_parallel_default    default number of queries run at once
## tpl_keywords            the quoted keyword names, or a command substitution
##                         that asks CASTEP for them
## tpl_guard               `$mufinder_guard ` when tpl_castep does not apply
##                         the time limit itself, otherwise nothing
## tpl_help_begin, tpl_help_end
##                         the marker lines around each description
## tpl_timeouts_tolerated  cut-off queries tolerated before giving up
## tpl_stopped_status      the exit status the loop gives up with
##
## Queries are started a few at a time where that pays, and their output is
## collected in files so that it is still printed whole and in order. Each group
## is printed as soon as it finishes, which keeps progress visible.
##
## Whether it pays depends on the node, so the loop measures it: it times one
## query alone, then starts a full group, and drops to one at a time unless the
## group finishes in clearly less time than its queries take in sequence. A
## group that overruns is abandoned at that point and its queries are run again
## in turn, so the trial costs a second or two at most. On SCARF a query mostly
## waited on MPI start-up, and four at once were much faster. On Redwood a lone
## query took 0.5 s but four sharing the job's single CPU took several seconds
## between them, slower than running them in turn. MUFINDER_PARALLEL sets the
## group size (1 disables grouping).
##
## The loop may run under `set -e` (login_command.sh), so every command that
## can fail harmlessly is followed by `|| true`.
##
## $mufinder_guard is the time limit on one query. A binary can answer
## --version and then hang in MPI_Init on a later start. Without a limit that
## would hold a strategy, and so the job, until the walltime. Every query is run
## under `timeout` where it exists, and the loop gives up once several have been
## cut off.
mufinder_guard=""
if command -v timeout >/dev/null 2>&1; then
  mufinder_guard="timeout ${MUFINDER_QUERY_TIMEOUT:-${tpl_query_timeout}}"
fi
mufinder_timeouts=0
## A -help query needs no network, but an MPI-built CASTEP still brings up its
## fabric first. On SCARF that cost about two of the 2.8 seconds each query
## took: Open MPI tried its OFI transport, failed, and fell back. Restricting
## Open MPI to its in-process transports skips that. The settings are applied
## only if CASTEP still answers with them, and other MPI libraries ignore them.
if [ "${MUFINDER_FAST_START:-1}" = 1 ] \
    && ${tpl_openmpi_settings} ${tpl_castep} --version 2>/dev/null | grep -q 'CASTEP version'; then
  export ${tpl_openmpi_settings}
fi
${tpl_timing}
mufinder_scratch=$(mktemp -d)
mufinder_parallel="${MUFINDER_PARALLEL:-${tpl_parallel_default}}"
## Nanosecond clock for the calibration below; BSD date prints a literal N, in
## which case calibration is skipped.
mufinder_now() { date +%s%N 2>/dev/null; }
case "$(mufinder_now)" in "" | *[!0-9]*) mufinder_calibrate=0 ;; *) mufinder_calibrate=1 ;; esac
[ "$mufinder_parallel" -gt 1 ] || mufinder_calibrate=0
mufinder_round=0
set -- ${tpl_keywords}
if [ "$#" -eq 0 ]; then echo "mufinder: CASTEP listed no keywords" >&2; fi
while [ "$#" -gt 0 ]; do
  ## While calibrating, the first round is a single query and the second a
  ## full group, so the two can be compared.
  size="$mufinder_parallel"
  if [ "$mufinder_calibrate" = 1 ] && [ "$mufinder_round" = 0 ]; then size=1; fi
  chunk=""
  count=0
  while [ "$#" -gt 0 ] && [ "$count" -lt "$size" ]; do
    chunk="$chunk $1"
    shift
    count=$((count + 1))
  done
  if [ "$mufinder_calibrate" = 1 ]; then mufinder_started=$(mufinder_now); fi
  pids=""
  for keyword in $chunk; do
    (
      status=0
      ${tpl_guard}${tpl_castep} -help "$keyword" > "$mufinder_scratch/$keyword.out" 2>&1 || status=$?
      printf '%s' "$status" > "$mufinder_scratch/$keyword.status"
    ) &
    pids="$pids $!"
  done
  if [ "$mufinder_calibrate" = 1 ] && [ "$mufinder_round" = 1 ] && [ "$count" = "$mufinder_parallel" ]; then
    ## The trial group. It is worth keeping only if it finishes in clearly less
    ## time than its queries take in turn; the moment it has used that much
    ## time it is abandoned rather than waited for (on Redwood such a group took
    ## 38 s against 0.5 s for one query).
    mufinder_limit_ns=$((mufinder_single * count * 3 / 4))
    while :; do
      pending=0
      for keyword in $chunk; do
        [ -f "$mufinder_scratch/$keyword.status" ] || pending=1
      done
      [ "$pending" = 1 ] || break
      mufinder_elapsed=$(( $(mufinder_now) - mufinder_started ))
      if [ "$mufinder_elapsed" -gt "$mufinder_limit_ns" ]; then
        echo "mufinder: $count queries at once had not finished after $((mufinder_elapsed / 1000000)) ms against $((mufinder_single / 1000000)) ms for one; running one at a time" >&2
        for pid in $pids; do
          ## Either may find nothing left to kill, or be missing; neither is
          ## an error, and the caller may be running under `set -e`.
          pkill -TERM -P "$pid" 2>/dev/null || true
          kill "$pid" 2>/dev/null || true
        done
        ## Anything that outlives the kill keeps writing into the old scratch
        ## directory, away from the queries that are run again.
        mufinder_abandoned="$mufinder_scratch"
        mufinder_scratch=$(mktemp -d)
        mufinder_parallel=1
        mufinder_calibrate=0
        set -- $chunk "$@"
        continue 2
      fi
      sleep 0.05
    done
  fi
  ## Only this round's queries: an abandoned one that survived being killed
  ## must not hold the loop up.
  wait $pids || true
  if [ "$mufinder_calibrate" = 1 ]; then
    mufinder_elapsed=$(( $(mufinder_now) - mufinder_started ))
    if [ "$mufinder_round" = 0 ]; then
      mufinder_single=$mufinder_elapsed
    else
      if [ "$count" = "$mufinder_parallel" ]; then
        echo "mufinder: $count queries at once took $((mufinder_elapsed / 1000000)) ms against $((mufinder_single / 1000000)) ms for one; keeping $count at a time" >&2
      fi
      mufinder_calibrate=0
    fi
    mufinder_round=$((mufinder_round + 1))
  fi
  for keyword in $chunk; do
    printf '${tpl_help_begin} %s@@\n' "$keyword"
    cat "$mufinder_scratch/$keyword.out"
    query_status=$(cat "$mufinder_scratch/$keyword.status")
    printf '\n${tpl_help_end} %s %s@@\n' "$keyword" "$query_status"
    ## 124 is how `timeout` reports that it cut the query off.
    if [ -n "$mufinder_guard" ] && [ "$query_status" = 124 ]; then
      mufinder_timeouts=$((mufinder_timeouts + 1))
    fi
  done
  if [ "$mufinder_timeouts" -ge ${tpl_timeouts_tolerated} ]; then
    echo "mufinder: CASTEP stopped responding: $mufinder_timeouts queries were cut off after ${MUFINDER_QUERY_TIMEOUT:-${tpl_query_timeout}} s each" >&2
    rm -rf "$mufinder_scratch" ${mufinder_abandoned:+"$mufinder_abandoned"}
    exit ${tpl_stopped_status}
  fi
done
rm -rf "$mufinder_scratch" ${mufinder_abandoned:+"$mufinder_abandoned"}
