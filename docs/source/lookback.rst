.. _lookback:

Lookback
########

In :code:`cuDART`, the use of lookback refers to performing a render while accounting for a finite speed of light. This means that when a ray samples a cell within a simulation domain,
it ensures that it reads in data from the proper snapshot in time, as appropriate for this position in space. For an observation made at :math:`t=t_\mathrm{obs}`, the time sampled along 
the ray path given by :math:`\boldsymbol{x} = \boldsymbol{x}_0 + s \hat{\boldsymbol{s}}` will be given by 

.. math::
    \bar{t} = t_\mathrm{obs} - \frac{|\boldsymbol{x} - \boldsymbol{x}_0|}{c} = t_\mathrm{obs} - \frac{s}{c}

For systems that are slowly evolving in time, this formulation is unimportant; all cells along the line of sight can be assumed to be sampled at the same time. However, for rapidly evolving 
systems, especially with velocities closely aligned to the line-of-sight, including this effect is incredibly important. Consider the following test problem, featuring two emitting regions travelling 
in opposite directions at :math:`\Gamma = 2`. These regions are spheres in their own rest frames, so are oblate spheroids with an axial ratio of :math:`\Gamma` due to Lorentz contraction. The minor axis
of the spheroid is aligned with its velocity. The figure below depicts the result of rendering this system when the velocity of the ejecta are aligned at :math:`\theta = \pi / 4`