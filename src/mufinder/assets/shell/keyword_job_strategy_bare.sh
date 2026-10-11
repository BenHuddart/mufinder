## Keyword job strategy when the cluster gives no launcher: run the CASTEP
## executable as it is. Part of keyword_job.sh.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. Lines starting ## are removed.
##
## tpl_help_loop_name  file name of the help loop script
## tpl_executable      the CASTEP executable
mufinder_try 'the CASTEP executable' bash ${tpl_help_loop_name} ${tpl_executable} || true
