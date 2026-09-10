# cuDART: CUDA + DDA Accelerated Ray Tracing (v1.0)

[![arXiv](https://img.shields.io/badge/arXiv-2609.09322-b31b1b.svg)](https://arxiv.org/abs/2609.09322)


`cuDART` (CUDA + DDA Accelerated Ray Tracing) is a toolkit designed for producing synthetic observations of optically thin emission. The code is designed to automatically account for relativistic and geometric effects such as beaming and light time delay. Running on the GPU and accelerated by the 3D-DDA (3D digitial differential analyzer) algorithm, `cuDART` is able to generate high-resolution observations from large simulations in seconds, facilitating easy comparison between numeric theory and real observations.

Documentation is hosted on [ReadTheDocs](https://cudart.readthedocs.io/en/latest/). You can also check out the code release paper on [arXiv](https://arxiv.org/abs/2609.09322), currently under review at [RAS Techinques and Instruments](https://academic.oup.com/rasti) journal.

This code is under active development, we recommend users obtain a stable release from the [Releases](https://github.com/hwhitehead/cuDART/releases). 

Gallery
-------

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/gallery/phenomena_comp.gif width="800" alt=animated/>
</p>
<p align="center"">
  <em> Comparison of renders of anti-parallel relativisitc ejecta, made using three different routines - without beaming/lookback, with beaming but no lookback and with both beaming and lookback. Here, lookback refers to accounting for the finite light time delay between emitter and observer. Each routine produces different observed behaviour, with only the last accurately recovering the expectation of relativistic and geometric theory. See the <a href="https://cudart.readthedocs.io/en/latest/phenomena.html">documentation</a> for a full discussion of these routines and results.</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/gallery/magnetised_jets.png width = "600"/>
</p>
<p align="center"">
  <em> Static images of a highly magnetised, variable power jet launched from an Active Galactic Nucleus, rendered from three different orientations by cuDART. Relativistic beaming results in a brighter advancing jet and dimmer receding jet; this effect is strongest when the jet is more closely aligned with the line-of-sight. Figure taken from <a href="https://ui.adsabs.harvard.edu/abs/2026arXiv260513469E/abstract"> Elley et al. 2026</a>.</em>
</p>

Installation
------------

This code is under active development, we recommend users obtain a stable release from the [Releases](https://github.com/hwhitehead/cuDART/releases). If you encounter any issues with the code or foresee avenues for improvement, please contribute to the Issues or Discussions sections, or get in contact with the development lead Henry Whitehead via henry[dot]whitehead[at]ist[dot]ac[dot]at.

The latest development version of `cuDART` can be cloned directly from this repository as 
```
$ git clone https://github.com/hwhitehead/cuDART/git
```
Once a local version of the codebase is available, the code can be built and executed by interacting with the Pythonic frontend, or directly with the C++/CUDA backend. See the documentation on ReadTheDocs (available online [here](https://cudart.readthedocs.io/en/latest/) and as source under `/docs/source`) for a full description of the API. 
