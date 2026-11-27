## Keyword job strategy for a job script that already runs inside a launcher
## step (addqueue). Part of keyword_job.sh.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. Lines starting ## are removed.
##
## tpl_standalone_seconds  seconds allowed for the first attempt
## tpl_launcher            the launcher words, each quoted
## tpl_help_loop_name      file name of the help loop script
## tpl_executable          the CASTEP executable
##
## addqueue does not run the job script as a batch script: it starts it with
## `srun --mpi=pmix`, so the script is already a task of an MPI job step. There
## a bare CASTEP is not an independent process but that step's rank 0; only its
## first start works, and when a later one fails in MPI_Init the scheduler
## cancels the whole step, script included (seen on Redwood: "STEP ...
## CANCELLED DUE TO TASK FAILURE", four seconds into the job). Starting CASTEP
## bare in the script is therefore never safe there. Launching every query
## through mpirun is safe and proven, but slow (Redwood: 2.4 s a query, over
## twenty minutes).
##
## So one launcher invocation is used as a *container*: `mpirun -np 1` runs the
## whole loop, and inside it the loop drops the variables that tie a process to
## an MPI job, so each CASTEP starts as a standalone single process as it would
## in a plain batch script, several at a time. If that goes wrong the damage is
## confined to that one mpirun, the dump is rejected, and the proven
## launcher-per-query path (one at a time, no Open MPI overrides) takes over.
mufinder_limit() {
  if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi
}
mufinder_try 'standalone CASTEP processes inside one launcher' mufinder_limit ${tpl_standalone_seconds} ${tpl_launcher} bash ${tpl_help_loop_name} --standalone ${tpl_executable} \
  || mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 MUFINDER_FAST_START=0 bash ${tpl_help_loop_name} ${tpl_launcher} ${tpl_executable} || true
