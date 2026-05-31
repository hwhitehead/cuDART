Intensity Calculation
#####################

.. _calculation_header:

:code:`cuDART` is designed to operate on simulation data with coordinates meshes defined in the observer frame (static Minkowski spacetime). 
While the code can be used to calculate other line-of-sight properties such as surface density, its real strength comes from 
calculating optically thin emergent intensities from domains populated with emissivity and velocity data.

The code is designed to accept emissivity data that is isotropic and defined in the cell's rest frame. To compute the 
emissivity in the observer frame, this emission must be beamed along the cell's direction of motion. Due to time dilation,
the frequency in the observer and emitter frames will also be different. :code:`cuDART` is able to account for all of these effects 
on the fly if it is provided with the monochromatic, isotropic rest-frame emissivity in each cell, the velocity in each cell, and the 
exponent for the assumed power-law frequency slope in the rest-frame emission. For a line of sight described by an origin :math:`\boldsymbol{x}_0` and normal :math:`\hat{\boldsymbol{s}}`, 
such that all points on the line of sight satisfy 

.. math::

    \boldsymbol{x} = \boldsymbol{x}_0 + s \hat{\boldsymbol{s}},

the emergent intensity :math:`I_{\nu}` measured at time :math:`t` can be computed as

.. math::

    I(\nu,\boldsymbol{x}_0,\hat{\boldsymbol{s}},t) &= \int j(\nu,\boldsymbol{x}) ds \\
                         &= \left(\frac{\nu}{\nu'_0}\right)^{\alpha}\int D(\boldsymbol{x},\hat{\boldsymbol{s}},\bar{t})^{2-\alpha} j'(\nu'_0,\boldsymbol{x},\bar{t})ds. 

Here :math:`j'` is the isotropic rest-frame emissivity, :math:`\nu'_0` is the monochromatic reference frequency, :math:`\alpha` is the exponent
for the power-law slope in the rest-frame emissivity and :math:`D` is the Doppler factor defined as 

.. math::

    D = \frac{1}{\gamma \left(1+\boldsymbol{\beta} \cdot \hat{s}\right)}, \quad \gamma = \frac{1}{\sqrt{1-\boldsymbol{\beta}\cdot \boldsymbol{\beta}}}

dependent on :math:`\boldsymbol{\beta} \equiv \boldsymbol{v}/c`, the local velocity in units of the speed of light. For a domain discretised into a 3D Cartesian mesh, this integration 
is calculated by summation 

.. math::

    I(\nu,\boldsymbol{x}_0,\hat{\boldsymbol{s}},t) = \left(\frac{\nu}{\nu'_0}\right)^{\alpha}\sum_n w_n D_n^{2-\alpha} j'_n(\nu_0,\bar{t})

where here the index :math:`n` labels cells on the line-of-sight with spatial indices :math:`\{i_n,j_n,k_n\}`. The weight factor
:math:`w_n` describes the length element of the line-of-sight within cell :math:`n`. The principle action of the DDA algorithm is to identify 
these cells on the line-of-sight, and their respective weights.

Note here that on the RHS of the equations above, all quantities are functions of :math:`\bar{t}`, not the observer time :math:`t`. This is because light takes a finite time
to travel from the emitter to the observer (e.g. :math:`\bar{t} \leq t`). The user can ignore this by setting :code:`lookback = False` at render, in which case 
the speed of light will be modelled as infinite and the above equation simplifies to :math:`\bar{t}=t`.

Lookback Time
-------------

If the speed of light is allowed to be finite, then the delay time between emission (at :math:`\bar{t}`) and observeration (at :math:`t`) 
will depend on the seperation between the cell and the observer such that

.. math::

    \bar{t} = t - \frac{s}{c}.

Suppose the simulation state is quantised in time at a fixed interval :math:`\Delta t`, and label successive physical state :math:`f` 
with the index :math:`m` such that

.. math::

    f_{n,m} = f_n(\bar{t}_m), \quad \bar{t}_m = m \Delta t 

For :math:`\bar{t} \neq m \Delta t`, we must apply linear interpolation between adjacent physical states. 
We define :math:`\bar{m} \equiv \text{floor}\left[\bar{t}/\Delta t\right]` as the index of the "early" timestep, and :math:`\delta_m(s,t)`
as the interpolation paramater between the true time :math:`\bar{t}` and the adjacent timestep :math:`\bar{t}_m`

.. math::

    \delta_m(s,t) = \frac{|\bar{t}-\bar{t}_m|}{\Delta t} = \frac{|t-\frac{s}{c}-m\Delta t|}{\Delta t}

We use this interpolation parameter to define the interpolation kernel :math:`W_{n,m} \in [0,1]`,

.. math::

    W_{n,m}(s,t) = \begin{cases}
    1 - \delta_m(s,t), & \text{if } m \in \left[\bar{m},\bar{m}+1\right] \\
    0, & \text{if } m \notin \left[\bar{m},\bar{m}+1\right]
    \end{cases}

The weighting kernel allows for an approximate sampling between intermediary states :math:`[\bar{m},\bar{m}+1]` for
:math:`\bar{t} \in [m\Delta t, (m+1)\Delta t]`, such that 

.. math::

    f_n(\bar{t}) \leftarrow W_{n,m}f_{n,\bar{m}} + W_{n,\bar{m}+1}f_{n,\bar{m}+1}

Hence, our full intensity calculation allowing for relativistic boosting and a finite speed of light takes the from

.. math::

    I(\nu, \boldsymbol{x}_0, \hat{\boldsymbol{s}},t) = \left(\frac{\nu}{\nu'_0}\right)^{\alpha} \sum_m \sum_n w_n W_{n,m} D^{2-\alpha}_{n,m} j'_{n,m}(\nu'_0)

where the first summation over :math:`m` describes multiple calls to the principle render routine acting on different snapshots :math:`m`,
and the second summation over :math:`n` describes the standard DDA integration routine, with an additional weighting :math:`W_{n,m}` that depends on both space and time.
For simulation data sampled at suitably high cadence, this method allows for the recovey of relativistic and geometric phenomena such as superluminal motion (see :ref:`here <phenomena>`).