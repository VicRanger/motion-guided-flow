Precomputed BRDF is used to demodulate lighting from scene color as in papers such as `FuseSR`[^fusesr] and `LMV`[^lmv].

We found this unnecessary and dont use it in our model, but we keep the asset `asset/precomputed_brdf_lut.exr` and code `utils/dataset_utils.py@L129-165` in the repo.



[^fusesr]: Zhong Z, Zhu J, Dai Y, et al. Fusesr: Super resolution for real-time rendering through efficient multi-resolution fusion[C]//SIGGRAPH Asia 2023 Conference Papers. 2023: 1-10.
[^lmv]: Wu Z, Zuo C, Huo Y, et al. Adaptive recurrent frame prediction with learnable motion vectors[C]//SIGGRAPH Asia 2023 Conference Papers. 2023: 1-11.