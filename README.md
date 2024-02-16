## [I Prefer Not To Say: Protecting User Consent In Models with Optional Personal Information](https://arxiv.org/abs/2210.13954) (AAAI-24)
**Preliminary Code Version**

This folder contains the code and the data sets to run the experiments described in the paper and our PUC-inducing data augmentation strategy.
To run this code, we only standard data science python packages are required (we tested with ``python=3.8.8``, ``pandas=1.3.3``, ``scikit-learn=0.24.1``, ``numpy=1.20.1``).

Experiments (subfolder ``experiments``):
  - The first experiment where Attributed Inference Restriction is studied (Table 3) is implemented in ``experiments/experiment1.py``. The performance metrics shown in Table 4a are also produced with this script.
  - The second experiment where we study the performance of PUC models under moltiple features (Table 4b) is implemented in ``experiments/experiment3real.py``.
  - The final experiment where the convergence behavior of PUCIDA is studies is implemented in ``experiments/experiment3synth.py`` (Figure 2) and ``experiments/experiment4synth.py`` (Figure 3).
  
  
Other important files:
  - ``utils/data_utils.py`` Implements data loaders and normalization, preprocessing.
  - ``utils/pucida.py`` Implementation of both the exhaustive and the stochastic PUC-inducing data augmentation (PUCIDA)
  - The original data sets that we used can be found in the folder ``data``
  
 Software versions used: . 
 
## Reference

If you find our code or ressources useful your own work, consider citing our paper, for instance with the following BibTex entry:
```
@inproceedings{leemann2024prefer,
  title={I Prefer not to Say: Protecting User Consent in Models with Optional Personal Data},
  author={Leemann, Tobias and Pawelczyk, Martin and Eberle, Christian Thomas and Kasneci, Gjergji},
  journal={AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2024},
  note={Accepted, to appear.}
}
```
 


