#!/usr/bin/env bash

# Echo in blue color. Actually, is yellow.
function blue() {
  es=`tput setaf 3`
  ee=`tput sgr0`
  echo "${es}$1${ee}"
}

# Check arguments
if [ "$#" -lt 1 ]; then
  blue "Illegal args. Usage: run-all.sh <cmd-config> <node ID>"
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
  blue "Killing python everywhere"
  sleep 3

	for ((i = 0; i < $process_idx; i++)); do
		name=${node_name_list[$i]}
    ssh -oStrictHostKeyChecking=no $name "sudo pkill python 1>/dev/null 2>/dev/null"
	done

	blue "Printing python processes that are still running..."

	for ((i = 0; i < $process_idx; i++)); do
		name=${node_name_list[$i]}
		ret=`ssh -oStrictHostKeyChecking=no $name "pgrep -x train"`
		if [ -n "$ret" ]; then
			blue "python still running on $name"
		fi
  done
fi

if [ "$#" -eq 2 ]; then
	name=${node_name_list[$2]}
  
  blue "Killing python on node $name"
  sleep 3
  ssh -oStrictHostKeyChecking=no $name "sudo pkill python 1>/dev/null 2>/dev/null"

	blue "Printing python processes that are still running"
	ret=`ssh -oStrictHostKeyChecking=no $name "pgrep -x train"`
	if [ -n "$ret" ]; then
		blue "python still running on $name"
	fi
fi

