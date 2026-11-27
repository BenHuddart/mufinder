## One command run on a cluster login node, passed to `/bin/bash -lc`.
##
## Template conventions (see mufinder.services.shell_templates): ${tpl_name} is
## a placeholder, filled in and shell-quoted by
## mufinder.services.cluster_keyword_generator; every other $ is the shell's
## own and is left alone. A placeholder alone on a line that is given no text
## removes the line. Lines starting ## are removed.
##
## tpl_cd          `cd <directory>`, or nothing
## tpl_shell_init  the cluster's environment setup, or nothing
## tpl_command     the command itself
set -e
source /etc/profile >/dev/null 2>&1 || true
${tpl_cd}
${tpl_shell_init}
${tpl_command}
