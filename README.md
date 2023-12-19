# PronyAC
This is a Python program for performing Prony Analytic Continuation.

The input of our simulations is an odd number of Matsubara points $G(i \omega_n)$ sampled on a uniform grid $\lbrace i\omega_{n_0}, i\omega_{n_0 + \Delta n}, \cdots, i\omega_{n_0 + (N_{\omega}-1) \Delta n} \rbrace$, where  $\omega_n=\frac{(2n+1)\pi}{\beta}$ for fermions and $\frac{2n\pi}{\beta}$ for bosons, $n_0 \geq 0$ is an integer controlling the number of the first few points we decide to discard (if any), $\Delta n \geq 1$ is an integer controlling the distance of successive sampling points, $N_{\omega}$ is the total number of sampling points and should be an odd number. 

To achieve best performance, please obey the following criteria: $n_0$ should be chosen as the smallest value so that $\min|i\omega_{n_0} - \xi_l|$ and $\max|i\omega_{n_0} - \xi_l|$ are of the same order and  function values between first two sampling points, i.e., $G(i\omega_{n_0})$ and $G(i\omega_{n_0 + \Delta n})$, do not change dramatically;  $N_\omega$ should be chosen to the value making mapped pole locations $\lbrace\tilde{\xi_l}\rbrace$  separated as far as possible; it is sufficient to set $\Delta n = \max\lbrace 1, \frac{\beta}{200} \rbrace$ for the 64-bit machine precision. Practically, our method is robust to variations in $N_\omega$ and relatively large $n_0$. So you may use a tentative $n_0$ and $N_\omega$, e.g., $n_0 = \frac{\beta}{10}$ and $N_\omega = 2001$, at the beginning to get an estimation of pole information, then use this information to choose optimal values of $n_0$ and $N_\omega$.

The Prony Analytic Continuation is performed using the following command:  
**PronyAC(G_w, w, optimize = True, n_k = 1001, x_min = -10, x_max = 10, y_min = -10, y_max = 10, err = None)**  
1. G_w is a 1-d array containing the Matsubara data.
2. w is the corresponding sampling grid.
3. n_k is the maximum number of contour integrals. Our method is not sensitive to this parameter, so you may leave it untouched.
4. Only poles located within the rectangle  x_min < x < x_max, y_min < y < y_max are recovered. Our method is not sensitive to these four paramters. So the range can be set to be relatively large.
5. If the error tolerance err is given, the continuation will be carried out in this tolerance; else if optimize is True, the tolerance will be chosen to be the optimal one;
else the tolerance will be chosen to be the last singular value in the exponentially decaying range.
6. For noisy data, it is highly suggested to set optimize to be True (and please do not provide err). Although the simulation speed will slow down (will take about several minutes), this will highly improve the noise resistance and the accuracy of final results. And to make this method as robust as possible, please use $N_\omega$ which makes the mapped poles as separate as possible.
7. For noiseless data, setting optimize to be True will still give best results. However, setting err=1.e-12 will be much faster and usually give nearly optimal results.
       
Other classes and functions are also provided to facilitate testings of toy models. The "examples" folder contains all original data of 
https://doi.org/10.48550/arXiv.2312.10576 and corresponding scripts to generate them. It might be a good start from there to get familiar with this program.
