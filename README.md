# AAND-Automatic-Author-Name-Disambiguation
The implementation code of the paper "Automatic Author Name Disambiguation by Differentiable Feature Selection"

* The code about the AND dataset is partially based on the repositories: https://github.com/allenai/S2AND#data

The usage:
1. Download the AND dataset by the code:
```
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/
```
2. Install the environment by
```
pip install -r requirements.in
```

3. Modify the dir variable in ``data/path_config.json`` and ``AAND.ipynb``

4. Run the jupyter notebook ``AAND.ipynb``
