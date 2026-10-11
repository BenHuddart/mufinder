## Keyword job strategy for a batch script with a launcher (`srun -n 1`): use
## the launcher as little as the cluster allows. Part of keyword_job.sh.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. Lines starting ## are removed.
##
## tpl_help_loop_name  file name of the help loop script
## tpl_launcher        the launcher words, each quoted
## tpl_executable      the CASTEP executable
## tpl_step_probe      quoted `bash -c` script that starts CASTEP twice
##
## Tried in order:
##
## 1. the bare executable, when it starts inside the allocation;
## 2. one job step that runs the whole loop, with the bare executable inside
##    it, when two consecutive starts inside a single step both work;
## 3. the launcher for every query, which mirrors how real jobs run and is the
##    fallback that always applies. One at a time: a one-task allocation cannot
##    run job steps side by side.
##
## The first two are attempted only when `timeout` exists: it bounds the probe,
## and inside the loop it bounds every query, so a binary that hangs instead of
## failing costs a strategy a few minutes rather than the job its walltime.
if ! command -v timeout >/dev/null 2>&1; then
  mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 bash ${tpl_help_loop_name} ${tpl_launcher} ${tpl_executable} || true
elif timeout 30 ${tpl_executable} --version 2>/dev/null | grep -q 'CASTEP version' \
    && mufinder_try 'the bare CASTEP executable' bash ${tpl_help_loop_name} ${tpl_executable}; then
  :
elif timeout 120 ${tpl_launcher} bash -c ${tpl_step_probe} 2>/dev/null | grep -q 'CASTEP version' \
    && mufinder_try 'a single job step' ${tpl_launcher} bash ${tpl_help_loop_name} ${tpl_executable}; then
  :
else
  mufinder_try 'the launcher for every query' env MUFINDER_PARALLEL=1 bash ${tpl_help_loop_name} ${tpl_launcher} ${tpl_executable} || true
fi
