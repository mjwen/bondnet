import os
import sys
import hypertunity as ht
from datetime import datetime

domain = ht.Domain(
    {
        # hgat layer
        "--num-gat-layers": {2, 3, 4},
        "--gat-hidden-size": {32, 64, 128},
        "--num-heads": {4, 8},
        # "--feat-drop":{0.0},
        # "--attn-drop":{0.0},
        # "--negative-slope":{0.2},
        "--residual": {1},
        # set2set layer
        "--num-lstm-iters": {6},
        "--num-lstm-layers": {3},
        # fc layer
        "--num-fc-layers": {2, 3, 4},
        "--fc-hidden-size": {32, 64, 128},
        # training
        "--gpu": {-1},
        "--lr": {0.001},
        "--weight-decay": {0.001, 0.01},
        "--batch-size": {100},
        "--epochs": {1000},
    }
)


# optimiser = ht.BayesianOptimisation(domain)
optimiser = ht.GridSearch(domain, sample_continuous=True)

reporter = ht.Table(domain, metrics=["score"])

n_steps = 1
batch_size = 162

with ht.Scheduler(n_parallel=batch_size) as scheduler:
    for i in range(n_steps):
        print(f"start step: {i} at {datetime.now()}")
        samples = optimiser.run_step(batch_size=batch_size, minimise=True)
        jobs = [
            ht.SlurmJob(
                task=os.path.join(os.getcwd(), "hgat_electrolyte_bonds.py"),
                args=s.as_dict(),
                meta={
                    "binary": "python",
                    "resources": {"cpu": 1, "time": "24:00:00"},
                    "extra": [
                        # "#SBATCH --partition=lr4",
                        # "#SBATCH --qos=condo_mp_lr2",
                        # "#SBATCH --account=lr_mp",
                        "#SBATCH --partition=lr4",
                        "#SBATCH --qos=lr_lowprio",
                        "#SBATCH --account=ac_mp",
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
