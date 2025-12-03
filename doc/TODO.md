TODO List   {#pageTODO}
=========
@todo
- A better way to handle various exchange correlation functionals
  - Initialization, finalizing phase
  - Category: (LDA, GGA, Hybrid, Meta, VdW) + (XCId / XId+CId). Or just
    always use XId + CId separate form (ABINIT might use this)
  - Need to make the same thing for DG
  - Design of XC class or something? (need discussion)
- Add support for libdbcsr and then for linear scaling? Or implement the
  native version?
  - Subspace problem in Chebyshev filtering. Use SCALAPACK with
    purification to handle the subspace problem. 
    o examples/diagonalize.cpp which has a SCALAPACK interface. Compare
      the performance of ScaLAPACK diagonalization and purification
      first for a fixed matrix.
  - LIBDBCSR format similar to PEXSI. Worth trying first. 
- OpenMP does not work for the new spinor multiplication due to the
  common dependence on the Fourier structure
  - The latest version is AddMultSpinorFineR2C, fft.inputVecR2C etc are
    shared between the threads, which limits the use of OpenMP. Perhaps
    increase the dimension of the buffer to the size of the local
    spinor, which allows the usage of FFTW_MANY or GURU interface in the
    future. In the case when number of threads equal to the number of
    local bands on the node, this implementation might be more advantageous.
  - Check with Weile on options
    1) OpenMP paralelization, multiple threads for a band
    2) add more buffer space for using thread parallelization for each
       band
   - Threaded FFTW can be tried with the new interface namely
     FFTWExecute
- Remove the unnecessary part of distinguishing coarse / fine grid
  (done), and
  document this a bit more properly. It is even possible to revamp the
  implementation by storing the wavefunction coefficients in the complex
  arithmetic (not sure)
  - When referring to the real space grid, there is no coarse grid,
    but only the fine real space grid. In the Fourier space, there are
    two grids. One is the same size as the real space grid, which is
    used for FFT. The second one is the Fourier grid restricted to a
    sphere, that is often used for storage of wavefunction coefficients,
    but nothing else.
  - In LOBPCG, the linear algebra is done with real arithmetic. If
    LOBPCG is to be performed for Fourier coefficients (like in KSSOLV),
    this LOBPCG part (linear algebra computatoin) need to be changed as
    well. This will eliminate the additional array which is Spinor in
    the real space with a coarse grid, and also requires eliminating the
    "coarse" in the code.
  - "coarse real space grid" might be useful for exchange calculation
    given experience from PEtot and VASP.
- Spinor class may be removed and moved to the Hamiltonian class. In
  the future different types of spinors should be treated with different
  classes of "Hamiltonian". The functions in spinor, such as
  preconditioners should also be moved to the Hamiltonian class (or
  KohnSham). (not sure still needed)
  - Hamitlonian should be a pure virtual class (does not implement
    anything actually)
  - Spin polarization should be kept in mind start from the beginning.
  - Spinor moved into Hamiltonian class. In KohnSham class, it is only
    for the spin-restricted calculation. Spinor has only two dimension,
    which is the number of grid points (in the Fourier space restricted to a
    sphere), and the number of bands. In spin-polarization, instead of
    having (ng,nc,nb) format, it might be better to have Spinor
    wavefun[numSpinComponent], where each wavefun has size (ng,nb).
    For the spin noncolinear polarization, the format of Spinor could be
    (ng,nc,nb) but this will be in the future. For k-point, the format is 
    Spinor wavefun[kpt]. Maybe k-point should be able to handle
    spin-polarization etc as well. But again in the future.
  - for eigensolver, in order to be compatible with multiple types of
    Hamiltonian classes, we need something like 
    LOBPCG( Hamiltonian, wavefun, param )
    where param is a simple struct to be declared in Hamiltonian class
    (or somewhere else) that is used to LABEL which part of the
    Hamiltonian class is used. This is immediately used when
    spin-polarization or k-point is needed.
  - In SCF, some "if statement" to detect the type of Hamiltonian should
    be needed. 
  - SCF, eigensolver supports multiple types of classes. Details in
    different realization of the Hamiltonian class.
  - Should be compatible with DGDFT. Currently DGDFT only supports
    KohnSham class essentially. It might be too difficult to support
    DGDFT for any class other than KohnSham at this point. Perhaps focus
    on hybrid functionals.
  - The new design should be combined with the design of spin
    polarization. This design instead should leave room for k-point
    implementation.
  - Super Hamiltonian class to handle spin?
  - 3/5/2016 Not sure Spinor class should be removed if we really only
    are going implement the spin case. Any extension to complex
    arithmetic (noncolinear spin and k-point seems to lead to very
    different code?)

- Hybrid is in the KohnSham class with sequential implementation.  
  - Parallel implementation for PWDFT (done)
  - Hybrid for DG

- Nonlocal pseudopotential format is not very compatible with GPU like
  structure

DONE recently
c Remove BLOPEX
t Remove SCALAR design (still to merge with the complex version)
c tentatively add core dumper 
  Better way to handle error: handling function taking a message as
  input is a more versatile way for handling error messaging. callstack
  procedure is slow and does not work for openmp. The DEBUG mode is too
  slow due to push/popstacks
  - Push/Popstack too expensive for simple operations. Throw error
    allows the error to be caught by the catch phrase, which gives the
    output of the callstack, but there is no way to use tools like gdb to
    look into the problem. Another simple way is to use `abort` but this
    implementation might be platform dependent and less informative. The
    third way is to rely on core dump, but not sure about massively parallel
    case.
  - try/catch and C++ exceptions not particularly useful for debugging.
  - coredumper
  - Encapsulate the error handling function (partially done)
c Combine PWDFT_bb and MD 
  c Should only have pwdft.cpp and dgdft.cpp
  c Standardize the output of initial and final results into subroutines
    that are shared between pwdft and dgdft. 
  c NVE, Nose-Hoover, BB, density extrapolation
  - into something called move_ions for geometry optimization and MD.
    o BFGS or preconditioned version (e.g. QE uses BFGS)
    o For 2000-10000 atoms, perhaps BFGS is too expensive. Maybe CG or
      FIRE alternative. PETOT uses CG?
    o MD: NVE, NH1, Langevin
c Refine Fourier to clean the normalization factors. Encapsulate the
  forward and backward Fourier transforms? In what convention?
  c Convention similar to QE can be considered.




12/6/2017: refactor2

@todo
c Merge refactor1 into master, and start a refactor2 branch (easy) done

c Test the output from testSpecies.C from qbox, and benchmark with the
  output from MATLAB and figure out the unit convention etc. done

c Adapt upf2qso to pspio.cpp/.hpp, and compare the results with MATLAB
  again. done

c Coulomb interaction etc should be revamped. qbox can help. done

- Add support from BigDFT's Poisson solver to evaluate the non-periodic
  boundary condition Coulomb. But perhaps the simplest is to find QE's
  treatment and use Makov-Payne type of correction first. This also
  outputs an estimate of the vacuum level. ESM / BigDFT's solver may be
  the right way to do things.


Code structure:

c esdf.cpp
  o add isUseVLocal to decide whether to use the VLocal formulation.  done
c utility.hpp
  o add additional component called VLocalSR for short range VLocal and
    its derivatives. done
c hamiltonian.cpp
  o CalculatePseudoPotential:  done
      if isUseVLocal == false
        do before
      else
        evaluate the short range VLocal contribution in pseudo.VLocalSR.  done
        evaluate the Gaussian pseudocharge contribution in pseudo.pseudoCharge  done.
  o CalculateVtot. done
      add contribution from VLocalSR
  o CalculateForce2: done.
      if isUseVLocal == false
        do before
      else
        Add contribution from Gaussian pseudocharge (similar to current)
        Add contribution from ionic VLocal
        Add contribution from short range repulsion

  o Cleanup: Merge later with CalculateForce done
  o Cleanup: remove the local pseudopotential and nonlocal
    pseudopotential stored on the coarse grid. This is not useful. done
      
c scf.cpp
  o add energySelf_ for the self energy, energySR_ and forceIonSR_ for
    the short range correction energy. done
  o CalculateEnergy: done
      if isUseVLocal == false
        do before
      else
        Add contribution of SR and self energy from hamiltonian.cpp

  o Perhaps include the contribution of short range and self energy  to
    energy / force in scf instead of Hamiltonian, since vdW etc
    is also included at the same level.  done 
  o May need to change the code from Harris energy etc. done
  o Merge the computation of external force and VdW force into
    CalculateForce? done

c periodtable.cpp
  o CalculateVLocalShortRange: 
    Reuse the Sparse structure and subtract contribution from Gaussian
    pseudocharge.  done
  o CalculateGaussianPseudoCharge. done
  o Read Gaussian pseudocharge. done
  o Setup.  Load the local pseudopotential, and modify the local part of the
    pseudopotential to remove the Gaussian pseudocharge. done 


Integrate with TDDFT branch:

c Unified treatment of efield at the Hamiltonian level. 

c Energy / force calculation all at the hamiltonian level. done

c Unified treatment of restarting / inputing position and velocity in
  TDDFT

 

Features included in PWDFT but may not in DGDFT:

- Support VLocal format and UPF format of pseudopotential

- Remove local and nonlocal pseudopotential on the coarse uniform grid.

- Calculate ionic repulsion, vdw, external energy / force in
  hamiltonian_dg rather than scf_dg

- Pay attention to the compatibility with the complex data type


Other functionalities in PWDFT:

c spline.h
  it is weird to have multiple spline routines. Need to get rid of it.

- Combine the PeriodTable and PeriodicTable class

c Need a MATLAB routine to visualize the locality / smoothness of the
  UPF file after subtracting the Gaussian pseudocharge contribution. 
  done. added upf_view

- PeriodTable: When the UPF format is stable enough, remove the support
  for binary file. This allows the cleanup of PTEntry / PTSample format.

- PeriodTable: add derivatives of the pseudopotential when needed (maybe
  in phonon calculation).  Smooth out the pseudopotential as QE does.
  This might smooth out the behavior of ONCV pseudopotential esp at high
  kinetic energy cutoff (still needs more testing)

- eigensolver.cpp: for HF molecule when ecut is large (80au)
  ~/ResearchBIN/dgdft/HF/PW_UPF/ecut
  the LOBPCGScaLAPACK version converges slower than LOBPCG with a single
  core. This might be a bug, or related to issues related to the
  deflation. Need to try Meiyue's more stable version of LOBPCG.
  PPCG seems to converge slower in general for this problem, and the
  number SCFs can also increase w.r.t. the number of processors.
  Maybe pseudopotential needs to be smoothed out?


- Many complex routines are very similar to the real version. Need a
  cleaner and more maintanable version.
  One possibility is to mimic qbox: store the wavefunction on the coarse
  Fourier grid, and add a label Spinor.IsReal() to indicate whether real
  arithmetic or complex arithmetic should be used.  This would condense 
  PPCG/LOBPCG Real/Complex versions. Just make sure that DGDFT changes
  accordingly as well.  Treatment of spin/k-point. Qbox's design is to
  have a SlaterDet class, and a Wavefunction class. The SlaterDet are
  stored as SlaterDets sd_[ispin][ikp] in wavefunction. This means spin
  and k-points are treated on different footing. It does not treat
  non-colinear spin.
