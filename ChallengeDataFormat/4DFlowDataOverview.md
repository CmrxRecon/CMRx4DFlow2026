# 4D Flow MRI Data Overview

This note describes the **4D Flow MRI data** reconstructed directly from acquired **rawdata**, its tensor shape, and how to compute **velocity** from the complex images.

---

## 1. Image Tensor Shape

The reconstructed complex image `img` has shape:

$$
(N_v,\; N_t,\; SPE,\; PE,\; FE)
$$

- **FE**: number of frequency-encoding samples  
- **PE**: number of phase-encoding samples  
- **SPE**: number of slice phase-encoding samples  
- **$N_t$**: number of cardiac phases (time frames)  
- **$N_v$**: number of velocity encodings  

Typically $N_v = 4$ (and in this challenge dataset, $N_v$ is always 4).

---

## 2. Complex Signal Model (Per Velocity Encoding)

Let $\mathrm{img}_v$ denote the complex image for velocity encoding index $v$.

### 2.1 Reference encoding ($v=0$)

$$
\mathrm{img}_0 = M_0\, e^{i\theta_0}
$$

- $M_0$ is the magnitude of $\mathrm{img}_0$
- $\theta_0$ is the **background phase** caused by system factors (e.g., field inhomogeneity)

### 2.2 Flow-encoded acquisitions ($v=1,2,3$)

$$
\mathrm{img}_v = M_v\, e^{i(\theta_0+\theta_v)}, \quad v\in\{1,2,3\}
$$

- $M_v$ is the magnitude of $\mathrm{img}_v$
- $\theta_v$ is the **velocity-induced phase** in direction $v$, typically wrapped to

$$
\theta_v \in [-\pi,\pi]
$$

Different velocity directions are encoded into the corresponding $\theta_v$.

---

## 3. Velocity Computation from Complex Images

Compute the phase difference relative to the reference encoding:

$$
\phi_v = \angle\!\left(\mathrm{img}_v \cdot \mathrm{conj}(\mathrm{img}_0)\right),
\quad v\in\{1,2,3\}
$$

Convert phase to velocity using the encoding-specific VENC:

$$
\mathrm{vel}_v
= \frac{\phi_v}{\pi}\,\mathrm{VENC}_v
= \frac{\angle\!\left(\mathrm{img}_v \cdot \mathrm{conj}(\mathrm{img}_0)\right)}{\pi}\,\mathrm{VENC}_v,
\quad v\in\{1,2,3\}
$$

Resulting range (per direction):

$$
\mathrm{vel}_v \in [-\mathrm{VENC}_v,\ \mathrm{VENC}_v]
$$

### Notes on `angle(·)` and `conj(·)`

- $\mathrm{conj}(\cdot)$ denotes the **complex conjugate**. For a complex number $z=a+ib$,  
  $$
  \mathrm{conj}(z)=a-ib
  $$
  Using $\mathrm{img}_v \cdot \mathrm{conj}(\mathrm{img}_0)$ cancels the shared/background phase term (approximately $\theta_0$), leaving mainly the velocity-induced phase contribution.

- $\angle(\cdot)$ denotes the **complex argument / phase** operator, i.e., it returns the phase of a complex number:
  $$
  \angle\!\left(re^{i\varphi}\right)=\varphi
  $$
  In practice, $\angle(\cdot)$ is computed via `atan2(imag, real)` and returns a **wrapped phase** in the principal range (commonly $(-\pi,\pi]$). Therefore $\phi_v$ is intrinsically wrapped.

---

## 4. Role of VENC

**VENC** (velocity encoding) is a scan parameter specifying the maximum unaliased velocity range.

Because the encoded phase $\theta_v$ is wrapped to approximately $[-\pi,\pi]$, VENC determines which physical velocity range is mapped onto that phase interval, leading to recovered velocities approximately in $[-\mathrm{VENC},\mathrm{VENC}]$ for each direction.

### VENC trade-off 

There is a fundamental trade-off between **velocity-to-noise** and **dynamic range**:

- **Higher VENC**  $\Rightarrow$ **lower velocity-to-noise** (reduced sensitivity): the same physical velocity produces a smaller phase shift, so noise in phase corresponds to larger uncertainty in velocity.
- **Lower VENC**   $\Rightarrow$ **higher velocity-to-noise** (increased sensitivity): the same physical velocity produces a larger phase shift, improving velocity precision.

However, if the true velocity magnitude exceeds the chosen VENC, the phase wraps (aliases), causing **velocity wrapping/aliasing**:

- when $|v| > \mathrm{VENC}$, the measured phase is wrapped back into $(-\pi,\pi]$, and the reconstructed velocity appears folded into $[-\mathrm{VENC},\mathrm{VENC}]$ unless additional phase unwrapping / anti-aliasing is applied.