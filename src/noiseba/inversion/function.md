
# Optimized Cost Function Design

The overall cost function is written as

$$
f = {\texttt{data\_misfit}} + \lambda \cdot {\texttt{model\_regularization}}
$$

where  
- $\lambda$ is the global regularization parameter that balances data fit and model complexity.

## 1. Data-misfit term

$$
{\texttt{data\_misfit}} = \frac{1}{M}\sum_{m=1}^{M}\omega_m\frac{1}{N_m}\sum_{n=1}^{N_m}\left(\frac{C_{mn}^{\text{obs}}-C_{mn}^{\text{syn}}}{\sigma_{mn}}\right)^2
$$

Variables  
- $C_{mn}^{\text{obs}}$: observed phase velocity for the $m$ th dispersion mode at frequency $n$  
- $C_{mn}^{\text{syn}}$: synthetic response  
- $\sigma_{mn}$: data uncertainty (default = $0.05\,C_{mn}^{\text{obs}}$)  
- $\omega_m$: user weight for the $m$ th dispersion mode 
- $N_m$: number of frequencies in $m$ th dispersion mode  
- $M$: total number of dispersion modes

## 2. Model-regularization term

$$
{\texttt{model\_regularization}} = \alpha_s\sum_{i=1}^{N_L}w_i^{\text{ref}}(V_{s,i}-V_{s,i}^{\text{ref}})^2 + \alpha_z\sum_{i=1}^{N_L-1}\left(\frac{V_{s,i+1}-V_{s,i}}{\Delta x_i}\right)^2 + \alpha_{zz}\sum_{i=1}^{N_L-2}\left(\frac{V_{s,i+2}-2V_{s,i+1}+V_{s,i}}{\Delta x_i^2}\right)^2
$$

Variables  
- $V_{s,i}$: shear-wave velocity in layer $i$  
- $V_{s,i}^{\text{ref}}$: reference-model velocity (optional)  
- $w_i^{\text{ref}}$: reference-model weight (optional)  
- $\Delta x_i$: thickness of layer $i$  
- $\alpha_s$: minimum-model penalty coefficient  
- $\alpha_z$: first-derivative penalty coefficient  
- $\alpha_{zz}$: second-derivative penalty coefficient  
- $N_L$: total number of layers

## 3. Regularization parameters

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | Global trade-off parameter |
| $\alpha_s$ | Strength of minimum-model term |
| $\alpha_z$ | Strength of first-derivative smoothness |
| $\alpha_{zz}$ | Strength of second-derivative smoothness |

All four parameters are user-supplied and ≥ 0.