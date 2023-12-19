# PronyAC
Python code for Prony Analytic Continuation

The input of our simulations is an odd number of Matsubara points $G(i \omega_n)$ sampled on a uniform grid 
\begin{align}
\{i\omega_{n_0}, i\omega_{n_0 + \Delta n}, \cdots, i\omega_{n_0 + (N_{\omega}-1) \Delta n} \},
\end{align}
where  $\omega_n=\frac{(2n+1)\pi}{\beta}$ for fermions and $\frac{2n\pi}{\beta}$ for bosons, $n_0 \geq 0$ is an integer controlling the number of the first few points we decide to discard (if any), $\Delta n \geq 1$ is an integer controlling the distance of successive sampling points, $N_{\omega}$ is the total number of sampling points and should be an odd number. 
We find that it is sometimes advantageous to choose $n_0$ different from $0$ (for fermions) or $1$ (for bosons). In this case, the final interpolant has to be validated at the discarded points to ensure that they are consistent with the interpolant to within $\varepsilon$.

To achieve best performance, we choose the  following heuristic criteria: $n_0$ should be chosen as the smallest value so that $\min|i\omega_{n_0} - \xi_l|$ and $\max|i\omega_{n_0} - \xi_l|$ are of the same order and 
function values between first two sampling points, i.e., $G(i\omega_{n_0})$ and $G(i\omega_{n_0 + \Delta n})$, do not change dramatically;  $N_\omega$ should be chosen to the value making $\{\tilde{\xi_l}\}$  separated as far as possible; it is sufficient to set $\Delta n = \max\{1, \frac{\beta}{200}\}$ for the 64-bit machine precision. Practically, our method is robust to variations in $N_\omega$ and relatively large $n_0$. 
