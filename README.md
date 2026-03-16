# PWDFT-DCU

PWDFT-DCU is a high-performance, scalable software package for first-principles electronic structure calculations and ab initio molecular dynamics (AIMD) simulations. Based on the open-source **PWDFT**, it is specifically ported and optimized for the **Deep Computing Unit (DCU)** heterogeneous platform on Sugon advanced supercomputing systems.

By solving the Kohn-Sham equations using a plane-wave basis set, PWDFT-DCU enables efficient simulations of both solid-state materials and molecular systems, comparable to software such as VASP and Quantum ESPRESSO.

## 🚀 Key Features

- **Advanced Acceleration Algorithms**: Integrated with state-of-the-art algorithms including **LOBPCG/PPCG** eigensolvers, **ACE** (Adaptively Compressed Exchange), **PCDIIS**, and **ISDF**. These implementations accelerate hybrid functional calculations by up to **100x** compared to traditional methods.
- **CPU-DCU Heterogeneous Computing**: Fully optimized for the Sugon DCU (GPU-like accelerator) using the **HIP** framework, delivering significant speedups over CPU-only execution.
- **World-Class Scalability**: Features a highly efficient parallel architecture. It supports simulations of systems with over **4,000 atoms** on **8,000+ CPU cores** and scales efficiently across **thousands of DCU cards**.

## 📁 Directory Structure

- `src/`: Core source code.
- `examples/`: Main program source and executable location.
- `config/`: Templates for compilation and input files.
- `include/`: Header files.
- `external/`: External libraries (e.g., `lbfgs`, `rqrcp`).
- `doc/`: Documentation and manuals.
- `utilities/`: Matlab scripts for post-processing and testing.

## 🛠 Installation & Compilation

### Prerequisites
Ensure the following dependencies are installed:
- **Compilers**: Intel C++/Fortran or GCC.
- **Libraries**: `FFTW-3.3.10`, `Libxc-6.2.2`, `Yaml-0.8.0`.
- **HPC Stack**: Sugon **DTK** (Device Training Kit, DCU software stack).

### Build Steps
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-repo/pwdft-dcu.git](https://github.com/your-repo/pwdft-dcu.git)
   cd pwdft-dcu
