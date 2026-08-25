.. _phenomena_header:

Captured Phenomena
##################

In supporting realtivistic beaming and a finite speed of light, cuDART is able to 
capture a wide range of relativstic and geometric effects that are crucial for comparing simulation
data with real observations. To demonstrate these effects, we utilise a mock simulation dataset consisting of two
homogenous emitters travelling at :math:`\Gamma=2` (equivalent to :math:`v \sim 0.9c`) in opposite directions (antiparallel). 
The emitters are spherical in their own rest frames, so in the lab-frame they are oblate spheroids with an axial ratio of :math:`1/\Gamma`.
In the figure below, we compare renders taken of this system when it is viewed at an angle :math:`\theta = \pi/4` to the advancing ejectum's 
direction of motion. The scene is rendered under three different treatments, showin the the three right panels:

1. Rendered without relativistic beaming, and without lookback
2. Rendered with relativistic beaming, but without lookback
3. Rendered with relativistic beaming and lookback

It is clear that the three treatments produce significantly different observations, we discuss the discrepancies and their origin in the sections below.

.. _phenomena_figure:

.. figure:: ../../gallery/phenomena_comp.gif
    :width: 800px

    Comparison of images rendered from mock data featuring relativistic anti-parallel ejecta imaged with/without relativistic beaming and with/without the lookback implementation. 
    Without lookback, rendering is performed using a single simulation snapshot. With lookback, multiple snapshots are scanned to account for a finite communication time between source and observer.
    In the left panel, the observed transverse motion for systems with/without lookback. In the right panels, the synthetic observations generated under different routines at an observer time given by the grey dashed line in the left panel. 
    Without beaming, the advancing and receding ejecta have the same brightness; when beaming is included the advancing ejecta is substantially brighter (shown by the flux ratio :math:`S^\mathrm{adv}_\nu/S^\mathrm{rec}_\nu`). 
    Without lookback, the ejecta exhibit symmetric transverse motion (:math:`\beta_\mathrm{T}\sim0.6`) and are imaged as oblate spheroids, coherent with their lab-frame morphology. 
    When lookback is included, the proper asymmetric transverse motion is captured, with the advancing ejectum appearing to move faster (:math:`\beta_\mathrm{T}\sim 1.6`) than the receding ejectum (:math:`\beta_\mathrm{T}\sim0.4`). 
    Further, the ejecta are observered as spheres, consistent with the relativistic/geometric predictions of the Penrose-Terrell effect (see :ref:`below <phenomena_deformation>`), and exhibit the proper flux ratio (see :ref:`below <phenomena_flux>`).

.. _phenomena_beaming:

Relativistic Beaming
--------------------

The left two images of the above :ref:`figure <phenomena_figure>` compare renders made using the rest frame emissivity or the emissivity beamed into the lab frame. 
In their own rest frames, the advancing and receding are identical, so when beaming is neglected the two ejecta exhibit the same brightness. 
Once beaming is included this symmetry is broken and the advancing ejectum is significant brighter due to the beaming of radiation toward the observer. 
Similarly, the receding ejecta is dimmer than the beaming-less case, as the emission is beaming beamed away from the observer. 
Comparing the flux (:math:`S_\nu \propto \int I_\nu dA`) emitted by the advancing and receding ejecta in the beamed case gives a ratio of :math:`\mathcal{S}\equiv S^\mathrm{adv}_\nu / S^\mathrm{rec}_\nu \sim40`, 
set by the :math:`D^{2-\alpha}` scaling that enters into the intensity integral (see :ref:`here <calculation_header>`). 
This is actually still the incorrect flux ratio, the true value is only recovered when lookback is also included, see :ref:`below <phenomena_flux>`.

.. _phenomena_superluminal:

Transverse Motion
-----------------
A textbook consequence of the finite travel time of light is a discrepancy between the true and observed transverse motion for an emitting region. 
This is most obvious when the true velocity of the region is directed partially toward the observer; because the distance between emitter and observer is decreasing, the observed transverse motion is larger than reality. 
For an object travelling at :math:`\beta = v/c` at an inclination of :math:`\theta` to the observer's line-of-sight, the apparent transverse velocity of the object :math:`\beta_\mathrm{T}` takes the form

.. math::

    \beta_\mathrm{T} =  \frac{\beta\sin\left(\theta\right)}{1-\beta\cos\left(\theta\right)}.

While the true velocity is constrained to :math:`\beta \in [0,1]` by relativity, the observer velocity is extremised w.r.t :math:`\theta` at :math:`\theta_\mathrm{crit}=\cos^{-1}(\beta)`; at this orientation :math:`\beta_\mathrm{T}=\Gamma \beta`. 
Hence, for :math:`\beta > 1/\sqrt{2}`, there exist orientations :math:`\theta \sim \theta_\mathrm{crit}` where :math:`\beta_\mathrm{T} > 1`. In this scenario, the object appears to be moving faster than the speed of light (termed superluminal motion). 
This result is only recoverable when a finite time delay is accounted for, hence synthetic observations which assume infinitesimal communication time between source and observer fail to report the proper transverse motion. 
The left panel of the above :ref:`figure <phenomena_figure>` shows the observed displacement of twin-ejecta moving at fixed velocity, with the right panels comparing renders made with and without lookback. 
Without lookback, both ejecta are observed to have the same transverse speed (dashed lines in left panel), but with lookback they exhibit the proper asymmetric motion with the approaching ejecta appearing to travel faster than the receding (solid lines in left panel). 

.. _phenomena_deformation:

Morphological Deformation
-------------------------

As discussed in the previous section, allowing for a finite light travel time between emitter and observer can result in different observed motions. 
Similarly, the difference in light travel time between the near and far surfaces of an emitting region can result in morphological differences between the emitter structure as measured in the lab-frame and as observed. 
A delay between the near and far emitting surfaces results in a observed deformation of the emitter's geometry along its direction of motion. 
As first discussed by `Penrose 1959 <https://ui.adsabs.harvard.edu/abs/1959PCPS...55..137P/abstract>`_ and `Terrell 1959 <https://ui.adsabs.harvard.edu/abs/1959PhRv..116.1041T/abstract>`_ (and hence known as the Penrose-Terrell effect), 
this deformation opposes the size change imparted by Lorentz contraction, resulting in the observed size of the region matching the measurement made in the emitter's rest frame. 
In the scenario discussed by Penrose and Terrell, an emitter that is spherical in its own rest frame, while Lorentz contracted in the lab frame, is observerd to be spherical due to the differential lag time between near and far surfaces of the sphere. 
An image of the sphere would appear to be rotated: in the limit of :math:`\beta \rightarrow 1`, the closest point on the sphere would appear to be the the most displaced along the sphere's direction of motion. 
In the above :ref:`figure <phenomena_figure>` we can see that rendering without lookback results in an image depicting (incorrectly), the lab-frame oblate spheroid structure. When lookback is included, the proper spherical observation is recovered.

.. _phenomena_flux:

Observed Flux
-------------

Each render in the above :ref:`figure <phenomena_figure>` also compares the ratio of flux between ejecta. A standard result for discrete emission is that the rest-frame flux and observer flux for a source of homogeneous velocity are related by

.. math::

    S_\nu = \int I_\nu d\Omega = \frac{D^{3-\alpha}}{L^2}\int j'_{\nu'}dV' \propto D^{3-\alpha}

where we have used the Lorentz invariance of :math:`I_\nu / \nu^3` and :math:`d\Omega=dA'/L^2` (the solid angle for a source at a distance :math:`L` to the observer): see `Lind et al. 1985 <https://ui.adsabs.harvard.edu/abs/1985ApJ...295..358L/abstract>`_ for a more detailed discussion. 
The ejecta travelling towards/away from the observer have identical structure in their own rest frames (labelled with primes), so the ratio of fluxes :math:`S_\nu` between advancing and receding ejecta should take the form

.. math::

    \mathcal{S}\equiv\frac{S_\nu^\mathrm{adv}}{S_\nu^\mathrm{rec}} = \left(\frac{D^\mathrm{adv}}{D^\mathrm{rec}} \right)^{3-\alpha} = \left(\frac{1+\beta \cos(\theta)}{1-\beta \cos(\theta)}\right)^{3-\alpha}

The flux ratio for all three cases is calculated by integrating over the pixels for the advancing and receding ejecta. When beaming is neglected, the ratio is simply unity as both ejecta are identical in the lab-frame. 
When beaming is included, but lookback is neglected the improper flux ratio is still incorrect; while on a cell-by-cell basis the emissivity has been properly boosted into the observer frame, by failing to track the proper emission morphology the total emergent flux has also been miscalculated. 
In contrast, when lookback is included, the ratio of fluxes matches the theoretical result to within :math:`0.2\%`. Note that it is also technically incorrect to apply :math:`D^{3-\alpha}` Doppler factors when integrating in the lab-frame, this scaling only holds if integration is performed in fluid rest frame. 
For a source of homogenous velocity, this introduces an erroneous overestimation for the emitter volume by a factor :math:`\gamma`. For more general relativistic fields, in which the velocity is unlikely to be homogeneous, there will not exist a coherant Doppler factor to convert between fluid and observer frames.

.. _phenomena_summary:

Summary
-------

It should be clear from discusssion above that while boosting the rest-frame emissivity to the lab frame is a requirement for imaging, alone it is insufficient to recover the full relativistic and geometric observational predictions. 
Only by including this beaming and accounting for a finite communication time between an emitting region and the observer can accurate synthetic observations be formed. This finite communication time, termed lookback in the cuDART framework, is included by default, requiring the user to provide a series of simulation snapshots in time.
The toy model used to demonstrate these discrepancies features an emitting region that is static in its own rest frame (the emissivity of the region does not change, the velocity is constant and the shape unchanged). In using this simple toy model, we can make direct comparison to known theoretical results for the expected motion, morphology and fluxes. 
In a less idealised astrophysical setting, none of these static properties are assured and there may exist no tractable analytical expectations. Such cases require numerical calculation to generate observations. 
It is important to note that in some systems it is reasonable to ignore the finite speed of light. If the morphology of a source (as defined in the lab frame), evolves slowly compared to its light self-crossing time, then the communication time between source and observer can be treated as effectively instantaneous and rendering can be performed on a snapshot-by-snapshot basis. 
In the language of `Lind et al. 1985 <https://ui.adsabs.harvard.edu/abs/1985ApJ...295..358L/abstract>`_, this is equivalent to treating the lab-frame as the pattern frame. This assumption is reasonable for some large-scale AGN jet structure, as the advance speeds of jets into the circum-galactic medium is usually much slower than the speed of light. 
However, caution is warranted when visualising rapidly evolving structures, such as knots in the jet beam, or comparing between the advancing and receding jets at late times as here the light time delay can become significant. Authentic visualisation of such structures will require schemes which account for a finite speed of light.

.. _phenomena_aliasing: 

Aliasing
--------

One source of morphological deformation along the line-of-sight that is *not* a physical consequence of a finite communication between the emitter and observer is aliasing. 
Aliasing occurs when the snapshots provided to the render routine have too large a seperation in time (too low a cadence). 
In this case, there may not be a suitably close simulation state for the render to sample when using the :code:`lookback` routine. 
This will result in artificial deformation, and can even result in multiple images of the same region
if the sampling cadence is low enough. A reasonable estimation for the critical sampling interval :math:`\Delta t_\mathrm{crit}` for resolving a region with characteristic
scale :math:`R` and velocity :math:`v` is the *observed* self-crossing time of the region, such that avoiding aliasing requires

.. math:: 

    \Delta t < \Delta t_\mathrm{crit} \equiv \frac{1 - \beta \cos(\theta)}{\sin(\theta)} \frac{R}{v}

Alternatively, for a given cadence of simulation data :math:`\Delta t`, the user can predict the smallest possible length scale resolvable as a function of :math:`v` and :math:`\theta`:

.. math::

    R_\mathrm{min} = \frac{v\sin(\theta)\Delta t}{1 - \beta \cos(\theta)}

The figure below compares renders of simulation data containing anti-parallel ejecta that are spheres in the lab frame. 
The renders read data sampled below, at and above the required frequency (left to right). It is clear that in the left panel,
the render is heavily aliased, with duplicate images of the emitting region appearing. 
Each individual image also depicts a spherical emitter, which is consistent with the lab-frame structure but is inconsistent with the spherical observation predicted by theory (see :ref:`discussion <phenomena_deformation>`).
This effect is diminished once the system is sampled at the critical frequency, with proper morphology mostly recovered but with edge artifacts persistent.
Above the critical frequency, the proper observed shape of the emitting region is faithfully recovered.

.. _phenomena_alias:

.. figure:: ../../gallery/alias.png
    :width: 800px