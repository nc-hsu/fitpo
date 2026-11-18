# References

De Luca, F., Vamvatsikos, D. and Iervolino, I. (2013). Near-optimal piecewise
linear fits of static pushover capacity curves for equivalent SDOF analysis.
*Earthquake Engineering & Structural Dynamics*, 42(4), 523–543.
<https://doi.org/10.1002/eqe.2225>

Used for: the fitting rules of {func}`fitpo.fit_deluca_backbone` and the elastic stiffness rule used by every fit.

Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2),
100–115. <https://doi.org/10.1093/biomet/41.1-2.100>

Used for: the CUSUM change detector in {func}`fitpo.fit_cbf_cusum_backbone`.

Ramer, U. (1972). An iterative procedure for the polygonal approximation of
plane curves. *Computer Graphics and Image Processing*, 1(3), 244–256.
<https://doi.org/10.1016/S0146-664X(72)80017-0>

Used for: line simplification in `fit_envelope(mode="simplified-loading-front")`.

Douglas, D. H. and Peucker, T. K. (1973). Algorithms for the reduction of the
number of points required to represent a digitized line or its caricature.
*The Canadian Cartographer*, 10(2), 112–122.
<https://doi.org/10.3138/FM57-6770-U75U-7727>

Used for: line simplification in `fit_envelope(mode="simplified-loading-front")`.
