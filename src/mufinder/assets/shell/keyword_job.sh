## The scheduler job that writes the keyword dump, modelled on the cluster's
## own job script.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. A placeholder alone on a line that is given no text
## removes the line. Lines starting ## are removed.
##
## tpl_preamble          the shebang and scheduler directives, plus the
##                       environment setup of the cluster's job template
## tpl_directory         the job directory
## tpl_shell_init        the cluster's keyword shell init, or nothing
## tpl_help_loop_name    file name of the help loop script written below
## tpl_help_loop_script  job_help_loop.sh
## tpl_log, tpl_dump     the job log and the dump
## tpl_dump_complete, tpl_help_begin, tpl_help_end
##                       the marker lines of the dump
## tpl_strategy          one of keyword_job_strategy_*.sh
##
## The strategy produces the dump by the fastest way that works. Starting
## CASTEP costs a fraction of a second, but starting a *job step* for it costs
## seconds: on SCARF, launching each of ~570 queries through `srun` ran at about
## 20 queries a minute. So the launcher is used as little as the cluster allows
## (keyword_job_strategy_job_steps.sh), except on addqueue clusters
## (keyword_job_strategy_launcher_step.sh) and where there is no launcher
## (keyword_job_strategy_bare.sh). Each attempt goes through mufinder_try,
## which keeps a dump only if it is complete and at least nine queries in ten
## succeeded; otherwise the next strategy overwrites it.
${tpl_preamble}
cd ${tpl_directory}
${tpl_shell_init}
cat > ${tpl_help_loop_name} <<'MUFINDER_HELP_LOOP'
${tpl_help_loop_script}
MUFINDER_HELP_LOOP
: > ${tpl_log}
mufinder_dump_ok() {
  grep -q '^${tpl_dump_complete}$' ${tpl_dump} || return 1
  total=$(grep -c '^${tpl_help_begin} ' ${tpl_dump})
  good=$(grep -c '^${tpl_help_end} [A-Z_0-9]* 0@@$' ${tpl_dump})
  [ "$total" -gt 0 ] && [ $((good * 10)) -ge $((total * 9)) ]
}
mufinder_try() {
  echo "mufinder: trying $1" >> ${tpl_log}
  shift
  "$@" > ${tpl_dump} 2>> ${tpl_log} && mufinder_dump_ok
}
${tpl_strategy}
