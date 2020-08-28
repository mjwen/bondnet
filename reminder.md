This is a reminder file for the admin of the repo.

# Update dependence conda package version

When changing the version of dependence conda packages, the below files should be updated:
- [README.md](./README.md)
- [pythonpackage.yml](./.github/workflows/pythonpackage.yml)
- [environment.yml](./binder/environment.yml)


# Update binder link to Jupyter notebook

When the Jupyter notebook used as demo on binder are replaced, the binder links in the
below files should be updated. 
- [README.md](./README.md)
- [predict/README.md](./bondnet/scripts/examples/predict/README.md)
- [train/README.md](./bondnet/scripts/examples/train/README.md)
