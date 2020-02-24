import os
import argparse
import numpy as np
import datetime
import sys
import sherpa
import time
import subprocess

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--num_hours', type=float,  default=0.1,
                    help='number of hours to run each trial')
parser.add_argument('--num_trials', type=int,  default=1000,
                    help='number of sherpa trials')
parser.add_argument('--expname', type=str,  default='tmp',
                    help='experiment name')
args = parser.parse_args()

def launch_trial(expname, num_hours, trial, host):

    y1 = trial.parameters["spline_y1"]
    y2 = trial.parameters["spline_y2"]
    y3 = trial.parameters["spline_y3"]

    #train_cmd="source ~/.bash_profile; killall python; zas; cd /users/ahjiang/src/Cutout;"
    train_cmd="source ~/.bash_profile; zas; cd /users/ahjiang/src/Cutout;"

    train_cmd += " python scripts/py/run_experiment_cutout.py"
    train_cmd += " --calculator=spline"
    train_cmd += " --strategy=sb"
    train_cmd += " --expname={}".format(expname)
    train_cmd += " --num-hours={}".format(num_hours)
    train_cmd += " --trial={}".format(trial.id)
    train_cmd += " --spline-y1={} --spline-y2={} --spline-y3={}".format(y1, y2, y3)

    print('============================================')
    print("[{}] Launching trial {} on {}".format(datetime.datetime.now(), trial.id, host))

    outfile = "/proj/BigLearning/ahjiang/output/cifar10/{}/sb-spline-alwayson_cifar10_wideresnet_0_128_1024_3_trial{}_seed1337_v5".format(expname, trial.id)

    ssh = subprocess.Popen(["ssh", "%s" % host, train_cmd],
                            shell=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return ssh, outfile

def main():
    parameters = [sherpa.Continuous(name="spline_y1", range=[0, 1]),
                  sherpa.Continuous(name="spline_y2", range=[0, 1]),
                  sherpa.Continuous(name="spline_y3", range=[0, 1])]

    expname = args.expname
    num_trials = args.num_trials
    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=num_trials)
    #algorithm = sherpa.algorithms.RandomSearch(max_num_trials=num_trials)

    stopping_rule = None
    study = sherpa.Study(parameters=parameters, algorithm=algorithm,
                         stopping_rule=stopping_rule, lower_is_better=False,
                         dashboard_port=8999)

    outdir = "/proj/BigLearning/ahjiang/output/cifar10/{}/".format(expname)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    summary_outfile = "{}/sb-spline-alwayson_cifar10_wideresnet_0_128_1024_3_trial0_seed1337_v5_summary".format(outdir)

    hosts=[#"h0.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h1.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h2.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h2.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h3.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h4.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h5.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h6.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h7.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h8.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h9.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h10.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h11.sb4-12.BigLearning.orca.pdl.cmu.edu",
           #"h0.sb4.BigLearning.orca.pdl.cmu.edu"]
           "h0.sb5.BigLearning.orca.pdl.cmu.edu"]

    sshs = {}
    outfiles = {}
    trials = {}

    for i, host in enumerate(hosts):
        trial = study.next()
        ssh, outfile = launch_trial(expname, args.num_hours, trial, host)
        sshs[i] = ssh
        outfiles[i] = outfile
        trials[i] = trial

    with open(summary_outfile, "w+") as f_summary:
        while True:
            for i, host in enumerate(hosts):
                result = sshs[i].stdout.readlines()
                outfile = outfiles[i]
                trial = trials[i]

                if result == []:
                    error = ssh.stderr.readlines()
                    print("ERROR", error)

                with open(outfile) as f:
                    for line in f:
                        if "test_debug" in line:
                            vals = line.rstrip().split(",")
                            loss = float(vals[4])
                            acc = float(vals[5])

                y1 = round(trial.parameters["spline_y1"], 3)
                y2 = round(trial.parameters["spline_y2"], 3)
                y3 = round(trial.parameters["spline_y3"], 3)

                line = "trial_result,{},{},{},{},{},{}".format(trial.id, y1, y2, y3, loss, acc)
                f_summary.write(line+"\n")
                f_summary.flush()
                print(line)

                if bool(study.get_best_result()):
                    best_trial_id = study.get_best_result()["Trial-ID"]
                    best_acc = study.get_best_result()["Objective"]
                    print("Best trial: {}, Acc:{}".format(best_trial_id, best_acc))

                study.add_observation(trial=trial, iteration=trial.id,
                                      objective=acc, context={'test_loss': loss, 'test_acc': acc})
                study.finalize(trial=trial)

                try:
                    trial = study.next()
                except StopIteration:
                    exit()

                ssh, outfile = launch_trial(expname, args.num_hours, trial, host)
                sshs[i] = ssh
                outfiles[i] = outfile
                trials[i] = trial
main()

