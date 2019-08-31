echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile
sleep 20

echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile
sleep 20

echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile
sleep 20

exit
echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test -b 128 -d cifar100 --profile
sleep 20

echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile
sleep 20

echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile
CUDA_VISIBLE_DEVICES=7 time python scripts/py/run_experiment_cutout.py -s sb -e test4 -b 128 -d cifar100 --profile
sleep 20


echo "--------------------------------------"
CUDA_VISIBLE_DEVICES=0 time python scripts/py/run_experiment_cutout.py -s sb -e test1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 time python scripts/py/run_experiment_cutout.py -s sb -e test2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 time python scripts/py/run_experiment_cutout.py -s sb -e test3 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=7 time python scripts/py/run_experiment_cutout.py -s sb -e test4 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=8 time python scripts/py/run_experiment_cutout.py -s sb -e test5 -b 128 -d cifar100 --profile 
