import os
import sys
import hypertunity as ht

domain = ht.Domain(
    {
        # model
        "--node-hidden-dim": {64},
        "--edge-hidden-dim": {128},
        "--num-step-message-passing": {6},
        "--num-step-set2set": {6},
        "--num-layer-set2set": {3},
        # training
        "--gpu": {-1},
        "--lr": {0.01},
        "--weight-decay": {0.0},
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
        samples = optimiser.run_step(batch_size=batch_size, minimise=True)
        jobs = [
            ht.SlurmJob(
                task=os.path.join(os.getcwd(), "mpnn_qm9.py"),
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
            print("sample: {}, evaluation: {}".format(s, e))
            sys.stdout.flush()
            reporter.log((s, e), meta="decide layer, could be output dir")

print(reporter.format(order="ascending"))
