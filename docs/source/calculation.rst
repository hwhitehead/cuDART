.. _calculation_header:

Intensity Calculation
#####################

cuDART is designed to operate on simulation data with coordinates meshes defined in the lab frame (a static Minkowski spacetime). 
While the code can be used to calculate other line-of-sight properties such as surface density, its primary use case is
calculating emergent intensities from optically thin systems, given a simulation domain populated with emissivity and velocity data.

The code is designed to accept emissivity data that is isotropic and defined in each cell's rest frame. The emission is then converted into the lab frame,
accounting for relativistic boosting and frequency changes due to time dilation.
cuDART performs this calculation on the fly, requiring only the monochromatic, isotropic rest-frame emissivity (for a generic referenc frenquency) in each cell, the velocity in each cell, and the 
exponent for the assumed power-law frequency slope in the rest-frame emission. 

For a line of sight described by an origin :math:`\boldsymbol{x}_0` and normal :math:`\hat{\boldsymbol{s}}`, 
all points along the specific line of sight satisfy 

.. math::

    \boldsymbol{x} = \boldsymbol{x}_0 + s \hat{\boldsymbol{s}}.

The monochromatic emergent intensity measured at time :math:`t` by an observer at :math:`\boldsymbol{x}_0` can be computed as

.. math::

    I_{\nu}(\boldsymbol{x}_0,\hat{\boldsymbol{s}},t) &= \int j_{\nu}(\boldsymbol{x}) ds \\
                         &= \left(\frac{\nu}{\nu'_0}\right)^{\alpha}\int D(\boldsymbol{x},\hat{\boldsymbol{s}},\bar{t})^{2-\alpha} j'_{\nu'_0}\left(\boldsymbol{x},\bar{t}(s)\right)ds. 

Here :math:`j'` is the isotropic rest-frame emissivity, :math:`\nu'_0` is the monochromatic reference frequency, :math:`\alpha` is the exponent
for the power-law slope in the rest-frame emissivity and :math:`D` is the Doppler factor defined as 

.. math::

    D = \frac{1}{\gamma \left(1+\boldsymbol{\beta} \cdot \hat{s}\right)}, \quad \gamma = \frac{1}{\sqrt{1-\boldsymbol{\beta}\cdot \boldsymbol{\beta}}}.

The Doppler factor is a function of :math:`\boldsymbol{\beta} \equiv \boldsymbol{v}/c`, the local velocity in units of the speed of light. For a domain discretised into a 3D Cartesian mesh, the intensity path integration 
is calculated by summation 

.. math::

    I_{\nu}(\boldsymbol{x}_0,\hat{\boldsymbol{s}},t) = \left(\frac{\nu}{\nu'_0}\right)^{\alpha}\sum_n \Delta s_n D_n^{2-\alpha} j'_{\nu_0',n}(\bar{t}_n)

where here the index :math:`n` labels cells intersecting the line-of-sight with spatial indices :math:`\{i_n,j_n,k_n\}`. The weight factor
:math:`\Delta s_n` describes the length element of the line-of-sight within cell :math:`n`. cuDART uses an adapted version of the 3DDDA (3D digital differential analyzer) algorithm to identify 
the indices of intersecting cells, and their respective weights. This reduces the complexity of the intersection test for a ray through a domain of size :math:`N^3` from :math:`O(N^3)` to :math:`O(N)`.

Note here that on the RHS of the equations above, all quantities are functions of :math:`\bar{t}_n`, not the observer time :math:`t_\mathrm{obs}`. This is because light takes a finite time
to travel from the emitter to the observer (e.g. :math:`\bar{t}_n \leq t_\mathrm{obs}`). The user can ignore this by setting :code:`lookback = False` at render, in which case 
the speed of light will be modelled as infinite and the above equation simplifies to :math:`\bar{t}_n=t_\mathrm{obs}\;\forall\;n`. The render can then be performed using a single snapshot. If lookback is enabled, then multiple snapshots must be read per render.

.. _calculation_lookback:

Lookback: Rendering with Light Delay
------------------------------------

If the speed of light is allowed to be finite, then the delay time between emission (at :math:`\bar{t}_n` for cell :math:`n`) and observation (at :math:`t_\mathrm{obs}`) 
will depend on the seperation between the cell and the observer such that

.. math::

    \bar{t} = t_\mathrm{obs} - \frac{s}{c}.

Suppose the simulation state is quantised in time at a fixed interval :math:`\Delta t`, and label successive physical states :math:`f` 
with the index :math:`m` such that

.. math::

    f_{n,m} = f_n(t_m), \quad t_m = m \Delta t 

For :math:`\bar{t}_n \neq m \Delta t`, we must apply linear interpolation between adjacent physical states. 
We define :math:`\bar{m} \equiv \text{floor}\left[\bar{t}/\Delta t\right]` as the index of the "early" timestep, and :math:`\delta_{n,m}(t_\mathrm{obs})`
as the interpolation parameter between the true local time :math:`\bar{t}_n` and the adjacent global discretised time :math:`\bar{t}_m`

.. math::

    \delta_{n,m}(t_\mathrm{obs}) = \frac{|\bar{t}_n-t_m|}{\Delta t} = \frac{|t_\mathrm{obs}-\frac{s_n}{c}-m\Delta t|}{\Delta t}.

We use this interpolation parameter to define the weighting kernel :math:`W_{n,m} \in [0,1]`,

.. math::

    W_{n,m}(t_\mathrm{obs}) = \begin{cases}
    1 - \delta_{n,m}(t_\mathrm{obs}), & \text{if } m \in \left[\bar{m},\bar{m}+1\right] \\
    0, & \text{if } m \notin \left[\bar{m},\bar{m}+1\right]
    \end{cases}

The weighting kernel allows the renderer to sample intermediary states with :math:`\bar{t}_n \in [m\Delta t, (m+1)\Delta t]` by interpolating between the adjacent states :math:`[f_{n,\bar{m}},f_{n,\bar{m}+1}]`
, such that 

.. math::

    f_n(\bar{t}_n) \leftarrow \sum_{m} W_{n,m}f_{n,m} = W_{n,\bar{m}}f_{n,\bar{m}} + W_{n,\bar{m}+1}f_{n,\bar{m}+1} \quad \mathrm{for}\;\bar{m}\equiv\left\lfloor\frac{\bar{t}_n}{\Delta t}\right\rfloor

Hence, our full intensity calculation allowing for relativistic boosting and a finite speed of light takes the from

.. math::

    I_{\nu}(\boldsymbol{x}_0, \hat{\boldsymbol{s}},t) = \left(\frac{\nu}{\nu'_0}\right)^{\alpha} \sum_m \sum_n \Delta s_n W_{n,m} D^{2-\alpha}_{n,m} j'_{\nu_0',n,m}

where the sum over :math:`n` describes a path summation through the domain (with line weighting :math:`\Delta s_n`) and the sum over :math:`m` combines contributions from each simulation snapshot (through the time/space dependent weighting :math:`W_{n,m}`).
For simulation data sampled at suitably high cadence (see a discussion of :ref:`aliasing <phenomena_aliasing>`), this method allows for the recovey of relativistic and geometric phenomena such as superluminal motion (see :ref:`here <phenomena_header>`).