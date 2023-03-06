# Wandb sweeps

In our paper, we train several hundred models with different configuration; we use the weights and biases sweeps feature to coordinate this. Begin by logging into weights and biases with `wandb login`. You can then initiate a sweep with:

```bash
wandb sweep sweeps/<SWEEP_NAME>.yaml
```

This instantiates a sweep controller hosted by weights and biases. You may then add agents which consume runs from the controller. Our code is configured for single GPU runs, so you can create agents with:

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent <USERNAME>/subgroup_separability/<SWEEP_ID>
```

Change the `CUDA_VISIBLE_DEVICES` variable to the index of the GPU you want to use. You can parallelise agents by running them on different machines, or by running multiple agents on the same machine with different GPU indices.

The sweep files just tell wandb to do a grid search over all combinations of the command line arguments in the file. If you want to run the sweeps manually or with another tool, simply adapt the command line arguments in the sweep files to your needs.
