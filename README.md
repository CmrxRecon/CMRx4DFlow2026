# CMRx4DFlow2026

## About Us
**Welcome to the 4D Flow MRI Reconstruction Challenge 2026 (CMRx4DFlow2026)**
—an integral part of the 29th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2026), hosted in Abu Dhabi, United Arab Emirates, from October 4th to 8th, 2026. LEARN MORE (MICCAI 2026 - 29. International Conference On Medical Image Computing & Computer Assisted Intervention)

[Website](https://cmrxrecon.github.io/2026) |
[Dataset](https://www.synapse.org/Synapse:syn64545434) |
[Publications](#publication-references)

![IntroImage](https://github.com/CmrxRecon/CMRx4DFlow2026/blob/main/Intro2026.png)

### Background

- **What 4D Flow MRI is:** A **3D + time** MRI technique that captures blood-flow dynamics across the full cardiac cycle. Beyond anatomy, it enables **quantitative** hemodynamic measures such as **flow velocity**, **wall shear stress (WSS)**, and **vorticity**—effectively creating a cardiovascular **digital twin**.
- **Clinical value:** Supports risk assessment for conditions like **aortic aneurysm**, **stenosis**, and **dissection**, and can aid **surgical planning/simulation**.

## Motivation

- **Key clinical pain point — scan time:** High-resolution 4D Flow acquisitions typically take **30–60 minutes**, increasing patient burden and motion artifacts, and reducing scanner throughput.
- **Current acceleration is still not enough:** Parallel imaging and standard compressed sensing often reduce scans to **10–20 minutes** (often at low-to-medium spatial resolution), still far from the desired **sub-5-minute** routine clinical window.
- **Unique opportunity for AI:** 4D Flow includes redundancy across **3D space**, **time**, and **multi-directional velocity encoding**, creating a large parameter space where **intelligent undersampling + AI reconstruction** can potentially deliver much higher acceleration.
- **Goal of the challenge:** Develop methods that reconstruct **high-fidelity flow and hemodynamic information** from **extremely undersampled data (≈10×–50× reduction)** to enable **ultra-fast, clinically viable** 4D Flow MRI.

<table class="cmr-table">
    <thead>
        <tr>
            <th>CMRxRecon Series</th>
            <th>Modalities</th>
            <th class="center">No. of Centers</th>
            <th class="center">No. of Scanners</th>
            <th>Populations</th>
            <th>Sampling Trajectory</th>
            <th class="center">No. of Subjects</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><span class="year y2023">2023</span></td>
            <td>Cine, Mapping</td>
            <td class="center">1</td>
            <td class="center">1</td>
            <td>Healthy individuals</td>
            <td>2D Uniform</td>
            <td class="subjects">~300</td>
        </tr>
        <tr>
            <td><span class="year y2024">2024</span></td>
            <td>Cine, T1 and T2 Mapping, Blackblood, Phase contrast, Tagging</td>
            <td class="center">1</td>
            <td class="center">1</td>
            <td>Healthy individuals</td>
            <td>2D Uniform; 3D k-t Uniform; 3D k-t Gaussian; 3D k-t Radial</td>
            <td class="subjects">~300</td>
        </tr>
        <tr>
            <td><span class="year y2025">2025</span></td>
            <td>Cine, T1, T2 and T2* Mapping; T1w; T2w; T1rho; Blackblood; Phase contrast; LGE; Perfusion</td>
            <td class="center">5+</td>
            <td class="center">10+</td>
            <td>Healthy individuals and patients with hypertrophic cardiomyopathy; dilated cardiomyopathy; myocardial infarction; coronary artery disease; arrhythmias, etc.</td>
            <td>2D/3D k-t Uniform; 2D/3D k-t Gaussian; 2D/3D k-t Radial</td>
            <td class="subjects">~600</td>
        </tr>
        <tr>
            <td><span class="year y2026">2026</span></td>
            <td>4D Flow MRI</td>
            <td class="center">10+</td>
            <td class="center">10+</td>
            <td>Healthy individuals; patients with multi-organ diseases (heart, brain, aorta, kidney, liver, carotid artery)</td>
            <td>3D Cartesian (k-t Gaussian)</td>
            <td class="subjects">~400</td>
        </tr>
    </tbody>
</table>

## Challenge tasks
To bridge the gap between research and clinical deployment, the **2026 challenge** utilizes over **400 cases** from **10+ centers** to evaluate reconstruction performance across four specific dimensions:

*   **Regular Task 1: Accurate Reconstruction under High Acceleration** – To evaluate the **robustness and generalization performance** of reconstruction models under **high (10x-50x) acceleration factors** across different clinical centers and various scanners.
*   **Regular Task 2: Fast Reconstruction under Limited Computing Resources** – To evaluate the **clinical performance and computational efficiency** of reconstruction models across **standardized hardware (NVIDIA A6000)**.
*   **Special Task 1: Generalizability across New Sites and Diseases** – To evaluate the **hardware robustness and cross-site generalization performance** of reconstruction models across different **magnetic field strengths (1.5T, 3T, and 5T)** and diverse clinical scenarios.
*   **Special Task 2: Generalizability across Different Anatomical Regions** – To evaluate the **universality and cross-organ generalization performance** of reconstruction models across multiple **anatomical regions** (e.g., brain, liver, kidney, and carotid arteries).
![TaskImage](https://github.com/CmrxRecon/CMRx4DFlow2026/blob/main/TaskImage2026.png)

## Package Structure

* `CMRx4DFlowReconDemo`: contains parallel imaging reconstruction code
* `CMRx4DFlowMaskGeneration`: contains code for varied undersampling mask generation
* `ChallengeDataFormat`: contains image quality evaluation code for validation and testing
* `Submission`: contains the structure for challenge submission



## Publication references

You are free to use and/or refer to the CMRx4DFlow2026 challenge and datasets in your own research after the embargo period (Dec. 2026), provided that you cite the following manuscripts:

**References of the CMRx Series Dataset**
1. Wang C, Lyu J, Wang S, et al. CMRxRecon: A publicly available k-space dataset and benchmark to advance deep learning for cardiac MRI. Scientific Data, 2024, 11(1): 687. Doi: https://doi.org/10.1038/s41597-024-03525-4 
2. Wang Z, Wang F, Qin C, et al. CMRxRecon2024: A Multimodality, Multiview k-Space Dataset Boosting Universal Machine Learning for Accelerated Cardiac MRI, Radiology: Artificial Intelligence, 2025, 7(2): e240443. Doi: https://doi.org/10.1148/ryai.240443
3. Wang Z, Huang M, Shi Z, et al. Enabling Ultra-Fast Cardiovascular Imaging Across Heterogeneous Clinical Environments with a Generalist Foundation Model and Multimodal Database. arXiv preprint arXiv:2512.21652, 2025. Doi: https://doi.org/10.48550/arXiv.2512.21652 

**CMRx Series Challenge Summary Papers**
1. Lyu J, Qin C, Wang S, et al. The state-of-the-art in cardiac MRI reconstruction: Results of the CMRxRecon challenge in MICCAI 2023. Medical Image Analysis, 2025, 101: 103485. Doi: https://doi.org/10.1016/j.media.2025.103485 
2. Wang K, Qin C, Shi Z, et al. Extreme cardiac MRI analysis under respiratory motion: Results of the CMRxMotion Challenge. Medical Image Analysis, 2025: 103883. Doi: https://doi.org/10.1016/j.media.2025.103883
3. Wang F, Wang Z, Li Y, et al. Towards Modality-and Sampling-Universal Learning Strategies for Accelerating Cardiovascular Imaging: Summary of the CMRxRecon2024 Challenge. IEEE Transactions on Medical Imaging, 2025. Doi: https://doi.org/10.1109/TMI.2025.3641610 

**Reference for previously algorithms from the organizers:**
1. Wang C, Li Y, Lv J, et al. Recommendation for Cardiac Magnetic Resonance Imaging-Based Phenotypic Study: Imaging Part. Phenomics. 2021, 1(4): 151-170. Doi: https://doi.org/10.1007/s43657-021-00018-x 
2. Lyu J, Li G, Wang C, et al. Region-focused multi-view transformer-based generative adversarial network for cardiac cine MRI reconstruction. Medical Image Analysis, 2023: 102760. Doi: https://doi.org/10.1016/j.media.2023.102760
3. Lyu J, Tian Y, Cai Q, et al. Adaptive channel-modulated personalized federated learning for magnetic resonance image reconstruction. Computers in Biology and Medicine, 2023, 165: 107330. Doi: https://doi.org/10.1016/j.compbiomed.2023.107330
4. Wang Z, Qian C, Guo D, et al. One-dimensional Deep Low-rank and Sparse Network for Accelerated MRI, IEEE Transactions on Medical Imaging, 42: 79-90, 2023. Doi: https://doi.org/10.1109/TMI.2022.3203312
5. Qin C, Schlemper J, Caballero J, et al. Convolutional recurrent neural networks for dynamic MR image reconstruction. IEEE transactions on medical imaging, 2018, 38(1): 280-290. Doi: https://doi.org/10.1109/TMI.2018.2863670
6. Lyu J, Wang S, Tian Y, et al. STADNet: Spatial-Temporal Attention-Guided Dual-Path Network for cardiac cine MRI super-resolution. Medical Image Analysis, 2024;94:103142. Doi: https://doi.org/10.1016/j.media.2024.103142
7. Wang Z, Xiao M, Zhou Y, et al. Deep separable spatiotemporal learning for fast dynamic cardiac MRI. IEEE Transactions on Biomedical Engineering, 2025.  Doi: https://doi.org/10.1109/TBME.2025.3574090 
8. Huang J, Yang L, Wang F, et al. Enhancing global sensitivity and uncertainty quantification in medical image reconstruction with Monte Carlo arbitrary-masked mamba. Medical Image Analysis, 2025, 99: 103334. Doi: https://doi.org/10.1016/j.media.2024.103334
9. Wang Z, Yu X, Wang C, et al. One for multiple: Physics-informed synthetic data boosts generalizable deep learning for fast MRI reconstruction. Medical Image Analysis, 2025, 103: 103616. Doi: https://doi.org/10.1016/j.media.2025.103616
10. Lyu J, Wang G, Wang Z, et al. Diffusion-prior based implicit neural representation for arbitrary-scale cardiac cine MRI super-resolution. Information Fusion, 2025: 103510. Doi: https://doi.org/10.1016/j.inffus.2025.103510

**References of the images cited in this website**
1. https://commons.wikimedia.org/w/index.php?curid=53001321.
2. Sandino, Christopher M., et al. Accelerated abdominal 4D flow MRI using 3D golden-angle cones trajectory. Proceedings of the Proc Ann Mtg ISMRM, Honolulu, HI, USA (2017): 22-27. 
3. Rice J, et al. In Vitro 4D Flow MRI for the Analysis of Aortic Coarctation. Proc. Intl. Soc. Mag. Reson. Med. 30 (2022): 0088.Doi: https://doi.org/10.58530/2022/0088
4. Peper, Eva S., et al. 10-fold accelerated 4D flow in the carotid arteries at high spatiotemporal resolution in 7 minutes using a novel 15 channel coil. Proceedings of the 24th Annual Meeting of ISMRM, Singapore. 2016.