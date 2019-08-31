echo "----------------cifar100----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py -s sb -s=nofilter -e 190830_lr1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=relative -e 190830_lr1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=random -e 190830_lr1 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f stale -c=relative -e 190830_lr1 -b 128 -d cifar100 --profile
date +%s
sleep 20

echo "----------------cifar10----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py -s sb -s=nofilter -e 190830_lr1 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=relative -e 190830_lr1 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=random -e 190830_lr1 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f stale -c=relative -e 190830_lr1 -b 128 -d cifar10 --profile
date +%s
sleep 20

echo "----------------svhn----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py -s sb -s=nofilter -e 190830_lr1 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=relative -e 190830_lr1 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f alwayson -c=random -e 190830_lr1 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py -s sb -s=sb -f stale -c=relative -e 190830_lr1 -b 128 -d svhn --profile
date +%s
sleep 20

echo "----------------cifar100 accelerated ----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=nofilter -e 190830_lr2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=relative -e 190830_lr2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=random -e 190830_lr2 -b 128 -d cifar100 --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f stale -c=relative -e 190830_lr2 -b 128 -d cifar100 --profile
date +%s
sleep 20

echo "----------------cifar10 accelerated ----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=nofilter -e 190830_lr1 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=relative -e 190830_lr2 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=random -e 190830_lr2 -b 128 -d cifar10 --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f stale -c=relative -e 190830_lr2 -b 128 -d cifar10 --profile
date +%s
sleep 20

echo "----------------svhn accelerated ----------------------"
date +%s
CUDA_VISIBLE_DEVICES=0 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=nofilter -e 190830_lr1 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=5 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=relative -e 190830_lr2 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=6 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f alwayson -c=random -e 190830_lr2 -b 128 -d svhn --profile &
CUDA_VISIBLE_DEVICES=7 python scripts/py/run_experiment_cutout.py --accelerate-lr -s sb -s=sb -f stale -c=relative -e 190830_lr2 -b 128 -d svhn --profile
date +%s
sleep 20
