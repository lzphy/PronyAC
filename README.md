# PronyAC
This is a Python program for performing Prony Analytic Continuation.

The input of the simulation is the Matsubara data $G(i \omega_n)$ sampled on a uniform grid $\lbrace i\omega_{n_0}, i\omega_{n_0 + \Delta n}, \cdots, i\omega_{n_0 + (N_{\omega}-1) \Delta n} \rbrace$, where  $\omega_n=\frac{(2n+1)\pi}{\beta}$ for fermions and $\frac{2n\pi}{\beta}$ for bosons, $n_0 \geq 0$, $\Delta n \geq 1$ and $N_{\omega}$ is the total number of sampling points.

The Prony Analytic Continuation is performed using the following command:

**PronyAC(G_w, w, err = None, optimize = False, symmetry = False, pole_real = False, reduce_pole = True, plane = None, k_max = 999, x_range = [-np.inf, np.inf], y_range = [-np.inf, np.inf])**
1. *G_w* is a 1-d array containing the Matsubara data.
2. *w* is the corresponding sampling grid $\lbrace \omega_n \rbrace$.
3. If *err* (>=1.e-12) is given, the continuation will be carried out in this tolerance.
4. If *err* is not given, the tolerance will be chosen to be the last singular value in the exponentially decaying range when *optimize* is False and will be chosen to be the presumably optimal one when *optimize* is True. There is no guarantee that setting *optimize* to be True will always find the optimal solution. So it is suggested to fine-tune *err* to achieve the best performance.
5. *symmetry* determines whether to impose the up-down symmetry in the complex plane.
6. *pole_real* determines whether to restrict the poles exactly on the real axis when *symmetry* is True.
7. *reduce_pole* determines whether to discard the poles whose weights are smaller than the error tolerance.
8. If *plane* is set to be not None, the computation of pole weights will be enforced in either the original plane (when *plane* is "z") or the mapped plane (when *plane* is "w").
9. *k_max* is the maximum number of contour integrals.
10. Only poles located within the rectangle  *x_range*[0] < x < *x_rang*[1] and *y_range*[0] < y < *y_range*[1] are retained.

Note: parameters 6 to 10 are rarely used in most cases. If there is no strong motivation to change them, please leave them as default.

The results are stored in variables *pole_weight* and *pole_location*. The instance method *check_valid()* is provided for checking intermediate steps.

Other classes and functions are also provided to facilitate testings of toy models. The "examples" folder contains all original data of 
https://doi.org/10.48550/arXiv.2312.10576 and corresponding scripts to generate them.
