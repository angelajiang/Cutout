#!/usr/bin/env bash

# Echo in blue color. Actually, is yellow.
function blue() {
  es=`tput setaf 3`
  ee=`tput sgr0`
  echo "${es}$1${ee}"
}

# Check arguments
if [ "$#" -ne 1 ]; then
  blue "Illegal args. Usage: check-all.sh <cmd-config>"
  exit
fi

cmd_config_file=$1
blue "Using config file = $cmd_config_file"

process_idx="0"

echo "Cmd config file:"
while read line; do
  node_name_list[$process_idx]=`echo $line | cut -d';' -f 1 | awk '{$1=$1};1'`
  cmd_list[$process_idx]=`echo $line | cut -d';' -f 2 | awk '{$1=$1};1'`

  echo $process_idx ${node_name_list[$process_idx]} ${cmd_list[$process_idx]}
  ((process_idx+=1))
done < $cmd_config_file
echo ""

# No argument => All nodes
if [ "$#" -eq 1 ]; then
  blue "Checking train.py everywhere"
	for ((i = 0; i < $process_idx; i++)); do
		name=${node_name_list[$i]}
    blue "Checking at #$name"
    ssh -oStrictHostKeyChecking=no $name "ps aux | grep ahjiang | grep train"
    ssh -oStrictHostKeyChecking=no $name "nvidia-smi"
    echo ""
    echo ""
	done
fi
