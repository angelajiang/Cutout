#!/usr/bin/env bash

# Check if a file ($1) exists. If it does not, exit.
function assert_file_exists() {
  if [ ! -f $1 ]; then
    echo "utils: File $1 not found! Exiting."
    exit 0
  fi
}

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


# Create autorun process arrays using the first $autorun_num_processes process
# names in autorun_process_file.
process_idx="0"

echo "Cmd config file:"
while read line; do
  node_name_list[$process_idx]=`echo $line | cut -d';' -f 1 | awk '{$1=$1};1'`
  cmd_list[$process_idx]=`echo $line | cut -d';' -f 2 | awk '{$1=$1};1'`
  echo $process_idx ${node_name_list[$process_idx]} ${cmd_list[$process_idx]}
  ((process_idx+=1))
done < $cmd_config_file
echo ""
# Here, process_idx = number of nodes

# Anaconda location depends on cluster
hostname=`hostname`
if [[ $hostname == *"narwhal"* ]]; then
  anaconda_src_cmd="source ~/src/anaconda2/bin/activate"
else
  anaconda_src_cmd="source ~/anaconda2/bin/activate"
fi


# No argument => All nodes
if [ "$#" -eq 1 ]; then
  blue "Starting commands on all machines"
  sleep 4

	for ((i = 0; i < $process_idx; i++)); do
		name=${node_name_list[$i]}
		cmd=${cmd_list[$i]}
		blue "Running on node $i, name $name, cmd $cmd"

		ssh -oStrictHostKeyChecking=no $name "\
      $anaconda_src_cmd; \
			cd src/Cutout/; \
      $cmd &" &
	done
fi

# One argument => One node
if [ "$#" -eq 2 ]; then
  name=${node_name_list[$2]}
  cmd=${cmd_list[$2]}
  blue "Running on node $2, name $name, cmd $cmd"

  ssh -oStrictHostKeyChecking=no $name "\
    source ~/src/anaconda2/bin/activate; \
    cd src/Cutout/; \
    $cmd &"
fi
