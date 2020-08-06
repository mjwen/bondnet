import sys
from pathlib import Path
import hypertunity as ht

domain = ht.Domain(
    {
        ## gated layers
        "--embedding-size": {24},
        "--gated-num-layers": {3, 4},
        "--gated-hidden-size": {64, 128, 192},
        "--gated-num-fc-layers": {2},
        "--gated-graph-norm": {0},
        "--gated-batch-norm": {1},
        "--gated-activation": {"ReLU"},
        # "--gated-residual": {1},
        "--gated-dropout": {0, 0.1},
        #
        ## readout layers
        # "--num-lstm-iters": {6},
        # "--num-lstm-layers": {3},
        ## fc layers
        "--fc-num-layers": {2},
        "--fc-hidden-size": {30},  # this will be ignored
        # "--fc-batch-norm": {0, 1},
        "--fc-activation": {"ReLU"},
        # "--fc-dropout": {0.0},
        #
        ## learning
        # "--lr": [0.001, 0.01],
        # "--weight-decay": {1e-6},
        # "--epochs": {1000},
        # "--batch-size": {100},
        "--restore": {1},
        #
        ## GUP usage
        # "--gpu": {0},
        # DDP
        "--distributed": {1},
        "--num-gpu": {2},
    }
)


# optimiser = ht.BayesianOptimisation(domain)
optimiser = ht.GridSearch(domain, sample_continuous=True)

reporter = ht.Table(domain, metrics=["score"])

n_steps = 1
batch_size = 12

with ht.Scheduler(n_parallel=batch_size) as scheduler:
    for i in range(n_steps):
        samples = optimiser.run_step(batch_size=batch_size, minimise=True)
        jobs = [
            ht.SlurmJob(
                task=Path.cwd().joinpath("gated_electrolyte_rxn_ntwk.py"),
                args=s.as_dict(),
                meta={
                    "binary": "python",
                    "resources": {"gpu": 2, "cpu": 4, "time": "240:00:00"},
                    "extra": [
                        ##############
                        # lowprior es1
                        #############
                        "#SBATCH --partition=es1",
                        "#SBATCH --qos=es_lowprio",
                        "#SBATCH --account=ac_mp",
                        #############
                        # lowprior lr3
                        #############
                        # "#SBATCH --partition=lr3",
                        # "#SBATCH --qos=lr_lowprio",
                        # "#SBATCH --account=ac_mp",
                        #############
                        ## condo lr4
                        #############
                        # "#SBATCH --partition=lr4",
                        # "#SBATCH --qos=condo_mp_lr2",
                        # "#SBATCH --account=lr_mp",
                        #############
                        ## lowprior lr4
                        #############
                        # "#SBATCH --partition=lr4",
                        # "#SBATCH --qos=lr_lowprio",
                        # "#SBATCH --account=ac_mp",
                        #############
                        ## condo lr6
                        #############
                        # "#SBATCH --partition=lr6",
                        # "#SBATCH --qos=condo_mp_lr6",
                        # "#SBATCH --account=lr_mp",
                        #############
                        ## cf1
                        #############
                        # "#SBATCH --partition=cf1",
                        # "#SBATCH --qos=condo_mp_cf1",
                        # "#SBATCH --account=lr_mp",
                        #############
                        #############
                        # "#SBATCH --mem=92G",
                        "#SBATCH --requeue",
                        "module load cuda",
                        "conda activate dgl",
                    ],
                },
            )
            for s in samples
        ]
        scheduler.dispatch(jobs)
        evaluations = [
            r.data for r in scheduler.collect(n_results=batch_size, timeout=None)
        ]
        optimiser.update(samples, evaluations)
        for s, e in zip(samples, evaluations):
            print(f"    sample: {s}, evaluation: {e}")
            sys.stdout.flush()
            # reporter.log((s, e), meta="decide layer, could be output dir")

# print(reporter.format(order="ascending"))
