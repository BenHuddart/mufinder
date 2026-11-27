## Records, on stderr, what one query costs on this node; part of help_loop.sh.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. Lines starting ## are removed.
##
## Generation time is start-up time multiplied by several hundred, and whether
## that start-up is spent computing or waiting decides what can speed it up.
## One timed query in the job log answers that without anyone having to
## reproduce the job's environment by hand.
##
## tpl_castep  the CASTEP invocation, launcher included
{
  echo "mufinder: $(nproc 2>/dev/null || echo '?') cpu(s) available, Open MPI fast start $([ -n "$OMPI_MCA_btl" ] && echo applied || echo 'not applied')"
  TIMEFORMAT='mufinder: one query took %R s (%U s user, %S s system)'
  time ${tpl_castep} -help TASK >/dev/null 2>&1 || true
} >&2
