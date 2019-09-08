python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=cifar10 -b 64 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=cifar10 -b 128 --dst-dir="/proj/BigLearning/dlwong/output"

# sb
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=cifar100 -b 64 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=cifar100 -b 128 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=nofilter -d=cifar10 -b 64 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=nofilter -d=cifar10 -b 128 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=nofilter -d=cifar100 -b 128 --dst-dir="/proj/BigLearning/dlwong/output"

# sbw2
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=svhn -b 64 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=sb -d=svhn -b 128 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=nofilter -d=svhn -b 64 --dst-dir="/proj/BigLearning/dlwong/output"
python scripts/py/run_experiment_cutout.py -e 190907_profile -s=nofilter -d=svhn -b 128 --dst-dir="/proj/BigLearning/dlwong/output"

