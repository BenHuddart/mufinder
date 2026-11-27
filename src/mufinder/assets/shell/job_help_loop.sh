## The keyword job's help loop: writes one complete dump to standard output.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. Lines starting ## are removed.
##
## tpl_mpi_job_environment_re  quoted grep -E pattern for the variables that
##                             tie a process to an MPI job
## tpl_version_begin, tpl_version_end
##                             the marker lines around `castep --version`
## tpl_help_loop               help_loop.sh, run through mufinder_castep
## tpl_dump_complete           the marker line that ends a complete dump
##
## Usage: bash <this script> [--standalone] <CASTEP command line...>
##
## It runs CASTEP as whatever command line it is given, so the same script
## serves every strategy in keyword_job_*.sh.
if [ "$1" = "--standalone" ]; then
  shift
  ## Forget the MPI job this shell belongs to, so that each CASTEP starts as
  ## an independent single process, exactly as it does in a plain batch script.
  for name in $(compgen -e | grep -E ${tpl_mpi_job_environment_re}); do
    unset "$name"
  done
fi
## $mufinder_guard is empty until the help loop sets the time limit.
mufinder_castep() { $mufinder_guard "${MUFINDER_CASTEP[@]}" "$@"; }
MUFINDER_CASTEP=("$@")
printf '${tpl_version_begin}\n'
mufinder_castep --version 2>&1 || true
printf '\n${tpl_version_end}\n'
${tpl_help_loop}
printf '${tpl_dump_complete}\n'
