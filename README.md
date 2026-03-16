# PWDFT-DCU

PWDFT-DCU is a high-performance, scalable software package for first-principles electronic structure calculations and ab initio molecular dynamics (AIMD) simulations. Based on the open-source **PWDFT**, it is specifically ported and optimized for the **Deep Computing Unit (DCU)** heterogeneous platform on Sugon advanced supercomputing systems.

By solving the Kohn-Sham equations using a plane-wave basis set, PWDFT-DCU enables efficient simulations of both solid-state materials and molecular systems, comparable to software such as VASP and Quantum ESPRESSO.

## 🚀 Key Features

- **Advanced Acceleration Algorithms**: Integrated with state-of-the-art algorithms including **LOBPCG/PPCG** eigensolvers, **ACE** (Adaptively Compressed Exchange), **PCDIIS**, and **ISDF**. These implementations accelerate hybrid functional calculations by up to **100x** compared to traditional methods.
- **CPU-DCU Heterogeneous Computing**: Fully optimized for the Sugon DCU (GPU-like accelerator) using the **HIP** framework, delivering significant speedups over CPU-only execution.
- **World-Class Scalability**: Features a highly efficient parallel architecture. It supports simulations of systems with over **4,000 atoms** on **8,000+ CPU cores** and scales efficiently across **thousands of DCU cards**.

## 📁 Directory Structure
After downloading and extracting the PWDFT-DCU package from GitHub, you will find the following main directories:
* `config/`: Example input files for compilation.
* `doc/`: Related documentation and user manuals.
* `examples/`: Source code for the main process and the generated executable.
* `external/`: External libraries (`lbfgs` and `rqrcp`).
* `include/`: Header files.
* `src/`: Core source program files.
* `utilities/`: MATLAB scripts for related testing and post-processing.

## 🛠 Installation & Compilation

### Prerequisites
Ensure the following libraries and environments are installed and properly configured:
* **Compilers**: Intel C/C++, GCC.
* **Libraries**: `fftw-3.3.10`, `libxc-6.2.2`, `yaml-0.8.0`.
* **HPC Toolkit**: **DTK** (the software stack for DCU, similar to CUDA for GPUs).

### Build Steps
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/HXTP/PWDFT-DCU]
   cd pwdft-dcu
2. **Set Environment Variables**:  
   Modify `env.sh` to correctly set the paths for `fftw`, `libxc`, `yaml`, `intel`, and `dtk`.
3. **Configure Compilation Settings**:  
   Modify `make.inc` and double-check all paths. 
   * Set `USE_GPU = 1` to compile the **DCU version**.
   * Set `USE_GPU = 0` to compile the **CPU version**.
4. **Compile**:  
   Run the compilation script:
   ```bash
   sh compile.sh
Upon success, the green executable `pwdft` will appear in the `examples/` directory.

---

## 📝 Usage

### Input Files
PWDFT-DCU uses a format similar to **Quantum ESPRESSO**:
* **config.yaml**: Main parameter file (atomic structure, methods, etc.).
* **Pseudopotentials**: Supports **HGH** and **ONCV** (`.upf` or `.bin`). Download ONCV from [mat-simresearch.com](http://www.mat-simresearch.com).

### Output Files
* **statfile.0**: Detailed log including SCF iterations, energy convergence, and atomic forces.
* **STRUCTURE**: Binary file containing final geometry (generated upon normal exit).

### Testing
A benchmark for **Si64** is provided in the `test/` directory, including input files and Sugon job submission scripts (`srun.sh`, `intelmpi_bind.sh`).

---

## 📚 Citations
Please cite the following papers if you use this software:

> [1] W. Hu, et al. *Adaptively compressed exchange operator...* **JCTC**, 13(3):1188–1198 (2017).  
> [2] W. Hu, et al. *Projected commutator diis method...* **JCTC**, 13(11):5458–5467 (2017).  
> [3] W. Hu, et al. *Interpolative separable density fitting...* **JCTC**, 13(11):5420–5431 (2017).  
> [4] K. Dong, et al. *Interpolative separable density fitting through centroidal voronoi tessellation...* **JCTC**, 14(3):1311–1320 (2018).  
> [5] W. Hu, et al. *Accelerating excitation energy computation...* **JCTC**, 16(2):964–973 (2020).  
> [6] L. Wan, et al. *Hybrid MPI and OpenMP parallel implementation...* **Electron. Struct.**, 3(2):024004 (2021).  
> [7] J. Feng, et al. *Massively parallel implementation of iterative eigensolvers...* **CPC**, 299:109135 (2024).
