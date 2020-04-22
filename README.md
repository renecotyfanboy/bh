# bh_imager

This module allows the simulation of schwarzschild black hole images. This task requires the 
integration of the trajectory of photons passing close to the event horizon, which is a 
computationally intensive task. In order to allow a fast processing, the code relies on the `numba` 
package for a pre-compilation of vectorial calculations, as well as the `adaptive` package to estimate 
relevant points regarding image variability.
