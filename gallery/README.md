# Gallery

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/gallery/superluminal.gif width="800" alt=animated/>
</p>
<p align="center"">
  <em> Animation showing synthetic radio observations of relativistic twin ejecta launched at angles of &pi/2 &pi/4 to the line-of-sight (left and right respectively). In both cases, each blob has the same absolute velocity, but appear to move differently. cuDART automatically accounts for relatiivsitc beaming (the ejectum pointed toward the observer is brighter) and superluminal motion (as the approaching ejectum travels towards the observer, its apparent transverse velocity exceeds the speed of light).</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/comp.gif width="800" alt=animated/>
</p>
<p align="center"">
  <em> Animation depicting multiple views for a jet launched from an Active Galactic Nucleus, showing images rendered with unboosted and boosted data. Top panel shows that as the orientation of the jet changes, the unboosted luminosity is fixed but the boosted luminosity varies. Inset panel depicts a real observation of Hercules A. Simulation data featured in <a href="https://ui.adsabs.harvard.edu/abs/2026MNRAS.tmp..127E/abstract">this paper</a>.</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/append/docs/magnetised_jets.png width = "600"/>
</p>
<p align="center"">
  <em> Static images of a highly magnetised, variable power jet launched from an Active Galactic Nucleus, viewed from three different orientations. Relativistic beaming results in a brighter advancing jet and dimmer receding jet; this effect is strongest when the jet is more closely aligned with the line-of-sight. Simulation data produced as part of paper currently in prep.</em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/rotate.gif width = "600" alt=animated/>
</p>
<p align="center"">
  <em> Animation of a supernova-jet simulation snapshot, featuring 200 viewpoints each yielding a 2048<sup>2</sup> image. Raw image data for each frame generated in ~150ms, during which 4 million rays were cast through a simulation domain hosting over 400 million (750<sup>3</sup>) cells. Simulation data featured in <a href="https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.4011G/abstract">this paper.</a></em>
</p>

<p align="center">
  <img src=https://github.com/hwhitehead/cuDART/blob/main/docs/profiling.png width = "600" alt=animated/>
</p>
<p align="center"">
  <em> Scaling tests varying the size of both the domain and image, for a single homogenous meshblock. For reasonable sized domains, render runtime (per frame) scales linearly with both the side length of the domain, and number of pixels in the image. Adding boosting to the calculation increases runtime by 0.5dex.</a></em>
</p>