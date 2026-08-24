# cuDART: CUDA + DDA Accelerated Ray Tracing (v0.9)

`cuDART` (CUDA + DDA Accelerated Ray Tracing) is a toolkit designed for producing synthetic observations of optically thin emission. The code is designed to automatically account for relativistic and geometric effects such as beaming and light time delay. Running on the GPU and accelerated by the 3DDA (3D digitial differential analyzer) algorithm, `cuDART` is able to generate high-resolution observations from large simulations in seconds, facilitating easy comparison between numeric theory and real observations.

The code is currently under developement, but available freely for use by others. If you encounter any issues with the code or foresee avenues for improvement, please contribute to the Issues or Discussions sections, or get in contact with the development lead Henry Whitehead via henry[dot]whitehead[at]ist[dot]ac[dot]at.

Documentation is hosted on ReadTheDocs. 

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/gallery/magnetised_jets.png width = "600"/>
</p>
<p align="center"">
  <em> Static images of a highly magnetised, variable power jet launched from an Active Galactic Nucleus, rendered from three different orientations by cuDART. Relativistic beaming results in a brighter advancing jet and dimmer receding jet; this effect is strongest when the jet is more closely aligned with the line-of-sight. Figure taken <a href="https://ui.adsabs.harvard.edu/abs/2026arXiv260513469E/abstract">from Elley et al. 2026</a>.</em>
</p>

Installation
------------

The latest version of `cuDART` can be cloned directly from this repository as 

  $ git clone https://github.com/hwhitehead/cuDART/git

Specific versions of the codebase are availble on the Releases page. 

Once a local version of the codebase is available, the code can be built and executed by interacting with the Pythonic frontend, or directly with the C++/CUDA backend. See the documentation on ReadTheDocs (available online and as source under `/docs/source`) for a full description of the API. 
