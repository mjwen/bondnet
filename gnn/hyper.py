import os
import sys
import hypertunity as ht

domain = ht.Domain(
    {
        "--num-gat-layers": {2, 3, 4},
        "--gat-hidden-size": {32, 64, 128},
        "--num-heads": {4, 8},
        # "--feat-drop":{0.0},
        # "--attn-drop":{0.0},
        # "--negative-slope":{0.2},
        "--residual": {1},
        "--num-fc-layers": {2, 3, 4},
        "--fc-hidden-size": {32, 64, 128},
        # "--lr": [0.001, 0.01],
        "--weight-decay": {0.001, 0.01},
        "--epochs": {10},
    }
)


# optimiser = ht.BayesianOptimisation(domain)
optimiser = ht.GridSearch(domain, sample_continuous=True)

reporter = ht.Table(domain, metrics=["score"])

n_steps = 2
batch_size = 2

with ht.Scheduler(n_parallel=batch_size) as scheduler:
    for i in range(n_steps):
        samples = optimiser.run_step(batch_size=batch_size, minimise=True)
        jobs = [
            ht.Job(
                task=os.path.join(os.getcwd(), "lithium.py"),
                args=s.as_dict(),
                meta={"binary": "python"},
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
