.. _development_header:

Future Development
##################

Development for cuDART is use-case driven, there is no point adding features to a code that no one will use. 
If you have a project in mind that you think would benefit from additions being made to the code, contact the dev team 
using `GitHub <https://github.com/hwhitehead/cuDART>`_ or by email. 
That said, there are multiple planned avenues for future improvement which we present here.

* :ref:`Polarisation and Isotropy <development_polarisation>`
* :ref:`Rest Frame Spectra <development_spectra>`
* :ref:`Accelerating File Reads <development_fread_accel>` 
* :ref:`Concurrent Rendering <development_concurrent>` 

.. _development_polarisation:

Polarisation and Isotropy
-------------------------

There are a number of additional physical processes that would be interesting to include within the cuDART toolkit. Perhaps the most obvious would be 
polarisation; in principle the polarisation state of emergent radiation could be computed by combining Stokes vectors along the ray path. Computing the 
local polarisation state would likely require information about the magnetic field structure; loading this data would impose additional overheads. 
However, given the importance of polarisation in observational astrophysics, including this information within the render could prove very useful.
It is possible that the code would struggle to produce realistic results, as including depth-dependent effects such as Faraday rotation would be difficult
within the current framework. Including magnetic field information could also be used to introduce anisotropy in the rest frame, implemented as

.. math::

    j'_{\nu'}(\boldsymbol{x},\hat{\boldsymbol{s}})= S(j'_{\nu'}(\boldsymbol{x}) ,\hat{\boldsymbol{s}},\boldsymbol{B}(\boldsymbol{x}) )

Currently, all emission is assumed to be isotropic in the rest frame of the bulk flow, this asssumption could be relaxed if directional dependence can be informed from the local magnetic field structure.

.. _development_spectra:

Rest Frame Spectra
-------------------

Currently the code accepts as input the monochromatic rest-frame emissivity at some reference frequency, and then assumes a power-law profile for emission in order to inform the emissivity
at other frequencies. More complex rest-frame spectra can be readily supported provided sensible analytical maps can be formed between the emissivity at the reference and some other frequency.
In principle, any analytical function satisfying the form

.. math::

    j'_{\nu'}(\boldsymbol{x}) = J(j'_{\nu'_0}(\boldsymbol{x}),\nu',\nu'_0)

can be readily included with minimal changes. Even in the case where emission is well described by a single-slope power law, there is no formal requirement for the slope :math:`\alpha` to be spatially homogeneous or static. 
If theoretical models for an adaptive :math:`\alpha=\alpha(\boldsymbol{x},t)` can be formed, then the code would be able to support dynamic rest-frame spectra. This would break the global power-law scaling in the resultant image, 
but could be useful for probing the age of different emitting regions by observing the cooling of electron populations.

.. _development_fread_accel:

Accelerating File Reads
-----------------------

The most significant bottleneck for all but the smallest of simulation domains is the cost of reading files from storage into host memory.
This results in a runtime that is very sensitive to the users storage environment and read speed. 
Some of this cost is unavoidable, but for datasets featuring a large number of snapshots there is the potential that cuDART will load files 
that do not make any contribution to the render, due to an absence of temporal overlap. Work is underway to develop tools (under the "flexload" umbrella) that will intelligently 
avoid wasted file loads and render calls. Minimising the total loaded data could be assisted by pre-partitioning the simulation domain into sub-domains labelled using a bounding volume hierachy,
allowing for rapid identification of relevant regions. However, such partitioning would incur its own pre-render overhead, so the utility of this approach is not as obvious.

.. _development_concurrent:

Concurrent Rendering
--------------------

Imaging rapidly evolving sources, especially when the viewing angle is closely aligned to the direction of motion, requires a very high cadence of simulation snapshots
in order to produce accurate synthetic observations. Retaining a large number of simulation snapshots in storage represents a significant burden. 
One method to avoid this overhead is to write high cadence snapshots, perform renders on the fly and then delete the majority of snapshots, leaving only 
low cadence writes to other analysis. Given the complexity of relativistic hydrodynamic simulations and the efficiency of the render process, there are no concievable situations where
the render would be unable to keep up with the simulation, provided GPU resources are consistently available. In principle the user can construct their own tools to enact this routine, but it would be useful to build generic routines
to tail simulation programmes and concurrently render the high cadence data. 
