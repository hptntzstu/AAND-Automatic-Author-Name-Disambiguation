# AAND-Automatic-Author-Name-Disambiguation
The implementation code of the paper "Automatic Author Name Disambiguation by Differentiable Feature Selection"

The usage:
1. Download the AND dataset by the code:
```
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/
```

2. Modify the dir variable in ``data/path_config.json`` and ``AAND.ipynb``

3. Run the jupyter notebook ``AAND.ipynb``
