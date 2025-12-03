/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Chris J. Pickard and Lin Lin

This file is part of DGDFT. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
 */
/// @file esdf.cpp
/// @brief Electronic structure data format for reading the input data.
/// @date 2012-08-10
// all parameters with the yaml format, modified by xmqin 20230605
//
#include "esdf.hpp"
#include "utility.hpp" 
#include "domain.hpp"
#include "periodtable.hpp"
#include "yaml-cpp/yaml.h"
#include <xc.h>

namespace dgdft{


// *********************************************************************
// Electronic structure data format
// *********************************************************************

//===============================================================
//
// Electronic structure data format
// ---------------------------------------------------------------
//
//                            e s d f
//                            =======
//
// Author: Chris J. Pickard (c)
// Email : cp@min.uni-kiel.de
// Place : kiel, Germany
// Date  : 5/6th august 1999
//
// Summary
// -------
//
// This module is designed to simplify and enhance the input of data into
// electronic structure codes (for example, castep). It works from a
// fashion, and input is independent of the ordering of the input
// file. An important feature is the requirement that most inputs require
// default settings to be supplied within the main program calling
// esdf. This means that rarely used variables will not clutter everyday
// input files, and, even more usefully, "intelligence" may be built into
// the main code as the defaults may be dependent of other set
// variables. Block data may also be read in. Another important feature
// is the ability to define "physical" values. This means that the input
// files need not depend on the internal physical units used by the main
// program.
//
// History
// -------
//
// Esdf has been written from scratch in f90, but is heavily based
// (especially for the concept) on the fdf package developed by alberto
// garcia and jose soler. It is not as "flexible" as fdf - there is no
// provision for branching to further input files. This simplifies the
// code, and I hope that it is still useful without this feature. Also,
// the input and defaults are not dumped to a output file currently. I've
// not found this a hindrance as of now.
//
// Future
// ------
//
// My intention is to make this release available to alberto garcia and
// jose soler for their comments. It might be a good idea to use this as
// a base for fully converting the fdf package to f90. Or it may remain
// as a cut down version of fdf. I certainly hope that a package of the
// fdf sort becomes widely used in the electronic structure community. My
// experience has been very positive.
//
// Usage
// -----
//
// First, "use esdf" wherever you wish to make use of its features. In
// the main program call the initialisation routine: call
// esdf_init('input.esdf'). "input.esdf" is the name of the input file -
// it could be anything. This routine opens the input file, and reads
// into a dynamically allocated storage array. The comments and blank
// lines are stripped out. You are now ready to use the
// esdf_functions. For example, if you want to read in the number of
// atoms in your calculation, you would use: natom =
// esdf_integer('numberofatoms',1), where 'numberofatoms' is the label to
// search for in the input file, and '1' is the default. Call esdf_close to
// deallocate the data arrays. You may then open another input file using
// esdf_init. It is not currently possible to open more that on input
// file at one time.
//
// Syntax
// ------
//
// The input file can contain comments. These are defined as anything to
// the right of, and including, '#', ';', or '!'. It is straightforward
// to modify the routine to accept further characters. Blank lines are
// ignored -- use comments and blank lines to make you input file
// readable.
//
// The "labels" are case insensitive (e.g. unitcell is equivalent to
// unitcell) and punctuation insensitive (unit.cell is equivalent to
// unit_cell is equivalent to unitcell). Punctuation characters are '.'
// and '-' at the moment. Again - use this feature to improve
// readability.
//
// The following are equivalent ways of defining a physical quantity:
//
// "Ageofuniverse = 24.d0 s" or "ageofuniverse : 24.d0 s" or
// "ageofuniverse 24.d0 s"
//
// It would be read in by the main program in the following way:
//
// Aou = esdf_physical('ageofuniverse',77.d0,'ns')
//
// "Aou" is the double precision variable, 77.d0 is the default number of
// "ns" or nanoseconds. 24s will be converted automatically to its
// equivalent number of nanoseconds.
//
// Block data should be placed in the input file as follows:
//
// Begin cellvectors
// 1.0 1.0 0.0
// 0.0 1.0 1.0
// 1.0 0.0 1.0
// end cellvectors
//
// And it may be read:
//
//   If(esdf_block('cellvectors',nlines))
//     if(nlines /= 3) then (... break out here if the incorrect number
// of lines)
//     do i=1,nlines
//       read(block_data(i),*) x,y,z
//     end do
//   endif
//
// List of functions
// -----------------
//
// Self explanatory:
//
// Esdf_string(label,default)
// esdf_integer(label,default)
// esdf_single(label,default)
// esdf_double(label,default)
// esdf_physical(label,default,unit)
//
// a little more explanation:
//
// Esdf_defined(label) is true if "label" found, false otherwise
//
// Esdf_boolean(label,default) is true if "label yes/true/t (case/punct.insens)
//                             is false if"label no/false/f (case/punct.insens)
//
// The help feature
// ----------------
//
// The routine "esdf_help(helpword,searchword)" can be used to access the
// information contained within the "esdf_key_mod" module.
//
// If "helpword" is "search" (case insensitive), then all variables whose
// description contains "searchword" will be output.
//
// If "helpword" is "basic", "inter", "expert" or "dummy" the varibles of
// that type will be displayed.
//
// If "helpword" is one of the valid labels, then a description of this
// label will be output.
//
// Finishing off
// -------------
//
// Two routines, "esdf_warnout" and "esdf_close", can be used to finish
// the use of esdf. "esdf_warnout" outputs esdf warnings to screen, and
// "esdf_close" deallocates the allocated esdf arrays.
//
// Contact the author
// ------------------
//
// This code is under development, and the author would be very happy to
// receive comments by email. Any use in a commercial software package is
// forbidden without prior arrangement with the author (Chris J. Pickard).
//---------------------------------------------------------------

namespace esdf{

// *********************************************************************
// Constants
// *********************************************************************
const int nphys = 57;
const int llength = 100;  /* length of the lines */
const int numkw = 500;   /* maximum number of keywords */



char **block_data;
int nrecords;
int nwarns;
char **llist;
char **warns;
char ***tlist;
char phy_d[nphys][11];          /* D - dimension */
char phy_n[nphys][11];          /* N - name */
double phy_u[nphys];            /* U - unit */

char kw_label[numkw][100];
int kw_index[numkw];
char kw_typ[numkw][4];

FILE *fileunit;

YAML::Node conf = YAML::LoadFile("config.yaml");// YAML input

// TODO: chenjunshi
#if 1
// for io comm
int nreader = 1, file_reader = 0;
MPI_Comm clustercomm;
#endif

/************************************************************ 
 * Main routines
 ************************************************************/
void esdf_key() {
  /*  ===============================================================
   *
   *   Module to hold keyword list. this must be updated as
   *   new keywords are brought into existence.
   *
   *   The 'label' is the label as used in calling the esdf routines
   *   'typ' defines the type, with the following syntax. it is 3
   *   characters long.
   *   the first indicates:
   *        i - integer
   *        s - single
   *        d - double
   *        p - physical
   *        t - string (text)
   *        e - defined (exists)
   *        l - boolean (logical)
   *        b - block
   *   the second is always a colon (:)
   *   the third indicates the "level" of the keyword
   *        b - basic
   *        i - intermediate
   *        e - expert
   *        d - dummy
   *
   *   'Dscrpt' is a description of the variable. it should contain a
   *   (short) title enclosed between *! ... !*, and then a more detailed
   *   description of the variable.
   *
   *  ===============================================================
   */

  int i=0;

  strcpy(kw_label[i],"atom_types_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"atom_type");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"atom_num");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"core_cutoff_radius");
  strcpy(kw_typ[i],"P:E");

  i++;
  strcpy(kw_label[i],"potential_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"local_component");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"move_flag");
  strcpy(kw_typ[i],"I:I");

  i++;
  strcpy(kw_label[i],"correlation_type");
  strcpy(kw_typ[i],"T:D");

  i++;
  strcpy(kw_label[i],"ion_energy_diff");
  strcpy(kw_typ[i],"P:I");

  i++;
  strcpy(kw_label[i],"atom_coord");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"grid_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"super_cell");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"upf_file");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"grid_spacing");
  strcpy(kw_typ[i],"P:E");

  i++;
  strcpy(kw_label[i],"miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"cheb_deg");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mixing_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"mixing_variable");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"mixing_param");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"mixing_maxdim");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"log_files");
  strcpy(kw_typ[i],"L:B");

  /* LL: Keywords added below for DGDFT */
  i++;
  strcpy(kw_label[i],"mixing_steplength");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"penalty_alpha");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_min_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eig_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"eig_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_inner_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_outer_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_outer_energy_tolerance");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"svd_basis_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"temperature");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"ion_temperature");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scf_inner_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_inner_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_outer_miniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_outer_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_energy_criteria_engage_ioniter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_outer_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_etot_diff");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"md_scf_eband_diff");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"dg_degree");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"alb_num");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"alb_num_element");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"scalapack_block_size");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"extra_electron");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mapping_mode");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"element_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"extra_states");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"periodtable");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"pseudo_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"output_dir");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"restart_mode");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"restart_density");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_wfn");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_density");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_potential");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_wfn");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_alb_elem_lgl");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_alb_elem_uniform");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"output_wfn_extelem");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_pot_extelem");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_eigvec_coef");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"output_hmatrix");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"element_position_start");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"element_grid_size");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"element_cell");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"position_start");
  strcpy(kw_typ[i],"B:E");



  i++;
  strcpy(kw_label[i],"max_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"buff_update");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"statfile");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"dg_solver");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"wallwidth");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"weightratio");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"eigperele");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"orbperele");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"input_format");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"extended_element_ratio");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"output_bases");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"basis_radius");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"atom_bohr");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"atom_ang");
  strcpy(kw_typ[i],"B:E");


  i++;
  strcpy(kw_label[i],"atom_red");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"buf_dual");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"vext");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"output_vtot");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"deltafermi");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"pw_solver");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"ppcg_sbsize");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"xc_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"vdw_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"calculate_aposteriori_each_scf");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"calculate_force_each_scf");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"potential_barrier");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_w");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_s");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"potential_barrier_r");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"ecut_wavefunction");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"density_grid_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"lgl_grid_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"gauss_interp_factor");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"gauss_sigma");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"periodize_potential");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"distance_periodize");
  strcpy(kw_typ[i],"B:E");

  i++;
  strcpy(kw_label[i],"solution_method");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"diag_solution_method");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"smearing_scheme");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"num_pole");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_distfft");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_scalapack");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_scalapack_pw");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_row_pexsi");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"num_proc_col_pexsi");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"num_proc_symb_fact");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"energy_gap");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"spectral_radius");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"matrix_ordering");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"inertia_count");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"inertia_count_steps");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"max_pexsi_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pexsi_method");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pexsi_npoint");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"mu_min");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_max");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"num_electron_pexsi_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_inertia_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_inertia_expansion");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"mu_pexsi_safeguard");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"unused_states");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"eig_tolerance_dynamic");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"ion_max_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"ion_move");
  strcpy(kw_typ[i],"T:E");


  i++;
  strcpy(kw_label[i],"geo_opt_max_force");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"geo_opt_nlcg_sigma");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"fire_nmin");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"fire_time_step");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"fire_atomic_mass");
  strcpy(kw_typ[i],"D:E");


  i++;
  strcpy(kw_label[i],"md_max_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_time_step");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_type");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_variable");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"md_extrapolation_wavefunction");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"thermostat_mass");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"langevin_damping");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"kappa_xlbomd");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"restart_position");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_velocity");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_position");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_velocity");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"output_xyz");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"scf_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"md_scf_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scf_phi_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_ace");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"hybrid_df_kmeans_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_kmeans_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_mu");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_gaussianrandom");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_num_proc_scalapack");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_df_tolerance");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"block_size_scalapack");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_active_init");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"hybrid_mixing_type");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"hybrid_ace_twice_pcdiis");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"exx_divergence_type");
  strcpy(kw_typ[i],"I:E");


  // Inputs related to Chebyshev polynomial Filtered SCF iterations for PWDFT    
  i++;
  strcpy(kw_label[i],"first_scf_pwdft_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"first_scf_pwdft_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"general_scf_pwdft_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_ppcg_use_scala");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_cheby_use_scala");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"pwdft_cheby_use_wfn_ecut_filt");
  strcpy(kw_typ[i],"I:E");


  // ~~**~~
  // Inputs related to Chebyshev ploynomial Filtered SCF iterations for DG
  i++;
  strcpy(kw_label[i],"diag_scfdg_by_cheby");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cheby_use_scalapack");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"first_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"first_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"second_scfdg_chebyouteriter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"second_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"second_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");



  i++;
  strcpy(kw_label[i],"general_scfdg_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"general_scfdg_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  // **###**
  // Inputs related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  i++;
  strcpy(kw_label[i],"scfdg_use_chefsi_complementary_subspace");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_chefsi_complementary_subspace_syrk");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_chefsi_complementary_subspace_syr2k");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_nstates");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cs_ioniter_regular_cheby_freq");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_cs_bigger_grid_dim_fac");
  strcpy(kw_typ[i],"I:E");


  // Inner LOBPCG related options
  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_lobpcgtol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_lobpcgiter");
  strcpy(kw_typ[i],"I:E");

  // Inner CheFSI related options
  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_use_inner_cheby");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_chebyfilterorder");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"scfdg_complementary_subspace_inner_chebycyclenum");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"use_atom_density");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"use_vlocal");
  strcpy(kw_typ[i],"I:E");

  // The following are for TDDFT
  i++;
  strcpy(kw_label[i],"tddft");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_auto_save_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"restart_tddft_step");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_ehrenfest");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_vext");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_dipole");
  strcpy(kw_typ[i],"I:E");


  i++;
  strcpy(kw_label[i],"tddft_vext_polx");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_poly");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_polz");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_freq");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_phase");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_amp");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_t0");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_tau");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_vext_env");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"tddft_method");
  strcpy(kw_typ[i],"T:B");

  i++;
  strcpy(kw_label[i],"tddft_max_iter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_delta_t");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_total_t");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_krylov_max");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_krylov_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_diis_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_phi_tol");
  strcpy(kw_typ[i],"D:E");

  i++;
  strcpy(kw_label[i],"tddft_phi_maxiter");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"tddft_diis_maxiter");
  strcpy(kw_typ[i],"I:E");

 #if 0
  // LRTDDFT options
  i++;
  strcpy(kw_label[i],"lrtddft");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"lrtddft_isdf");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"iptype_lrtddft_isdf");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i],"eigensolver_lrtddft");
  strcpy(kw_typ[i],"T:E");

  i++;
  strcpy(kw_label[i], "numproceigensolver_lrtddft");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i],"output_excitation_energy");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i],"output_excitation_wfn");
  strcpy(kw_typ[i],"I:E");

  i++;
  strcpy(kw_label[i], "nvband");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i], "ncband");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i], "nkband");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i], "nummufac_lrtddft_isdf");
  strcpy(kw_typ[i], "D:E");

  i++;    
  strcpy(kw_label[i], "numgaussianrandomfac_lrtddft_isdf");
  strcpy(kw_typ[i], "D:E");

  i++;    
  strcpy(kw_label[i], "tolerance_lrtddft");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "maxiterkmeans_lrtddft_isdf");
  strcpy(kw_typ[i], "I:E");
    
  i++;
  strcpy(kw_label[i], "tolerancekmeans_lrtddft_isdf");
  strcpy(kw_typ[i], "D:E");
  
  i++;
  strcpy(kw_label[i], "eigmaxiter_lrtddft");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i], "eigmintolerance_lrtddft");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "eigtolerance_lrtddft");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "outstartband");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i], "outendband");
  strcpy(kw_typ[i], "I:E");

  i++;
  strcpy(kw_label[i],"spectrum");
  strcpy(kw_typ[i],"I:E");
 
  i++;
  strcpy(kw_label[i], "lrtddft_vext_polx");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "lrtddft_vext_poly");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "lrtddft_vext_polz");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "lrtddft_omega_grid");
  strcpy(kw_typ[i], "D:E");

  i++;
  strcpy(kw_label[i], "lrtddft_sigma");
  strcpy(kw_typ[i], "D:E");

#endif

}

void esdf() {
  /*  we allow case variations in the units. this could be dangerous
   *  (mev --> mev !!) in real life, but not in this restricted field.
   *
   *  m - mass l - length t - time e - energy f - force p - pressure
   *  c - charge d - dipole mom - mom inert ef - efield
   */
  struct {
    char d[11];
    char n[11];
    double u;
  } phy[nphys]={
    {"m",   "kg",           1.0},
    {"m",   "g",            1.0e-3},
    {"m",   "amu",          1.66054e-27},
    {"l",   "m",            1.0},
    {"l",   "nm",           1.0e-9},
    {"l",   "ang",          1.0e-10},
    {"l",   "bohr",         0.529177e-10},
    {"t",   "s",            1.0},
    {"t",   "ns",           1.0e-9},
    {"t",   "ps",           1.0e-12},
    {"t",   "fs",           1.0e-15},
    {"e",   "j",            1.0},
    {"e",   "erg",          1.0e-7},
    {"e",   "ev",           1.60219e-19},
    {"e",   "mev",          1.60219e-22},
    {"e",   "ry",           2.17991e-18},
    {"e",   "mry",          2.17991e-21},
    {"e",   "hartree",      4.35982e-18},
    {"e",   "kcal/mol",     6.94780e-21},
    {"e",   "mhartree",     4.35982e-21},
    {"e",   "kj/mol",       1.6606e-21},
    {"e",   "hz",           6.6262e-34},
    {"e",   "thz",          6.6262e-22},
    {"e",   "cm-1",         1.986e-23},
    {"e",   "cm^-1",        1.986e-23},
    {"e",   "cm**-1",       1.986e-23},
    {"f",   "N",            1.0},
    {"f",   "ev/ang",       1.60219e-9},
    {"f",   "ry/bohr",      4.11943e-8},
    {"l",   "cm",           1.0e-2},
    {"p",   "pa",           1.0},
    {"p",   "mpa",          1.0e6},
    {"p",   "gpa",          1.0e9},
    {"p",   "atm",          1.01325e5},
    {"p",   "bar",          1.0e5},
    {"p",   "mbar",         1.0e11},
    {"p",   "ry/bohr**3",   1.47108e13},
    {"p",   "ev/ang**3",    1.60219e11},
    {"c",   "c",            1.0},
    {"c",   "e",            1.602177e-19},
    {"d",   "C*m",          1.0},
    {"d",   "D",            3.33564e-30},
    {"d",   "debye",        3.33564e-30},
    {"d",   "e*bohr",       8.47835e-30},
    {"d",   "e*ang",        1.602177e-29},
    {"mom", "kg*m**2",      1.0},
    {"mom", "ry*fs**2",     2.1799e-48},
    {"ef",  "v/m",          1.0},
    {"ef",  "v/nm",         1.0e9},
    {"ef",  "v/ang",        1.0e10},
    {"ef",  "v/bohr",       1.8897268e10},
    {"ef",  "ry/bohr/e",    2.5711273e11},
    {"ef",  "har/bohr/e",   5.1422546e11},
    {"e",   "k",            1.38066e-23},
    {"b",   "t",            1.0},
    {"b",   "ry/mu_bohr",   2.350499e5},
    {"b",   "g",            1.0e4}
  };
  int i;

  for (i=0;i<nphys;i++) {
    strcpy(phy_d[i],phy[i].d);
    strcpy(phy_n[i],phy[i].n);
    phy_u[i]=phy[i].u;
  }
}

/*   --------------  esdf_bcast ----------------------  */
/*   Modified by Lin Lin, Nov 9, 2010                   */
void esdf_bcast(){
  int  mpirank;
  int  mpisize;

  // TODO: chenjunshi
#if 0
  MPI_Comm world = MPI_COMM_WORLD;
#else
  MPI_Comm world = clustercomm;
#endif

  MPI_Comm_rank( world, &mpirank );
  MPI_Comm_size( world, &mpisize );

  const int MASTER = 0;
  int i, j;
  MPI_Bcast(&nrecords, 1, MPI_INT, MASTER, world);
  MPI_Bcast(&nwarns,   1, MPI_INT, MASTER, world);
  if( mpirank != MASTER ){
    esdf();
    esdf_key();
    block_data=(char **)malloc(sizeof(char*)*nrecords);
    llist=(char **)malloc(sizeof(char*)*nrecords);
    warns=(char **)malloc(sizeof(char*)*nrecords);
    tlist=(char ***)malloc(sizeof(char**)*nrecords);
    for (i=0;i<nrecords;i++) {
      block_data[i]=(char *)malloc(sizeof(char)*llength);
      llist[i]=(char *)malloc(sizeof(char)*llength);
      warns[i]=(char *)malloc(sizeof(char)*(llength+1));
      tlist[i]=(char **)malloc(sizeof(char*)*llength);
      for (j=0;j<llength;j++)
        tlist[i][j]=(char *)malloc(sizeof(char)*llength);
    }
  }
  for (i=0;i<nrecords;i++) {
    MPI_Bcast(block_data[i], llength, MPI_CHAR, MASTER, world);
    MPI_Bcast(llist[i],      llength, MPI_CHAR, MASTER, world);
    MPI_Bcast(warns[i],    llength+1, MPI_CHAR, MASTER, world);
    for (j=0;j<llength;j++)
      MPI_Bcast(tlist[i][j],    llength, MPI_CHAR, MASTER, world);
  }
}


/*   --------------  esdf_init  ----------------------  */
void esdf_init(const char *fname) {
  /* Initialize */
  const int ncomm=3;
  const int ndiv=3;
  int unit,ierr,i,j,k,ic,nt,ndef,nread,itemp,itemp2;
  char cjunk[llength],ctemp[llength];
  char comment[ncomm],divide[ndiv];
  bool inblock;
  char filename[llength];

  strcpy(filename,fname);

  /* Define comment characters */
  comment[0]='#';  divide[0]=' ';
  comment[1]=';';  divide[1]='=';
  comment[2]='!';  divide[2]=':';

  esdf();
  esdf_key();

  /* "Reduce" the keyword list for comparison */
  for (i=0;i<numkw;i++) {
    strcpy(ctemp,kw_label[i]);
    strcpy(kw_label[i],esdf_reduce(strlwr(ctemp)));
  }

  /* initializing the array kw_index */
  for (i=0;i<numkw;i++) kw_index[i]=0;

  /* Open the esdf file */
  // esdf_file(&unit,filename,&ierr); // No need to read dgdft/pwdft.in 
  // strcpy(cjunk,"Unable to open main input file \"");
  // strcat(cjunk,trim(filename));
  // strcat(cjunk,"\"");

  // if (ierr==1) {
  //   printf("ESDF WARNING: %s - using defaults",trim(cjunk));
  //   nread=0;
  // } 
  // else
  //   nread=INT_MAX;

  // TODO: comment till the end

  /* Count the number of records (excluding blank and commented lines) */
  // nrecords=0;

  // for (i=0;i<nread;i++) {
  //   getaline(fileunit,cjunk);
  //   for (j=0;j<ncomm;j++) {
  //     ic=indexch(cjunk,comment[j]);
  //     if (ic>0) {
  //       for (k=ic-1;k<llength;k++) cjunk[k]=' ';
  //       cjunk[llength-1]='\0';
  //     }
  //   }
  //   if (len_trim(cjunk)>0) nrecords++;
  //   if (feof(fileunit)) break;
  // }
  // rewind(fileunit);

  /* Allocate the array to hold the records and tokens */
  // block_data=(char **)malloc(sizeof(char*)*nrecords);
  // llist=(char **)malloc(sizeof(char*)*nrecords);
  // warns=(char **)malloc(sizeof(char*)*nrecords);
  // tlist=(char ***)malloc(sizeof(char**)*nrecords);
  // for (i=0;i<nrecords;i++) {
  //   block_data[i]=(char *)malloc(sizeof(char)*llength);
  //   llist[i]=(char *)malloc(sizeof(char)*llength);
  //   warns[i]=(char *)malloc(sizeof(char)*(llength+1));
  //   tlist[i]=(char **)malloc(sizeof(char*)*llength);
  //   for (j=0;j<llength;j++)
  //     tlist[i][j]=(char *)malloc(sizeof(char)*llength);
  // }

  /* Set the number of warnings to zero */
  // nwarns=0;
  // for (i=0;i<nrecords;i++) {
  //   for (j=0;j<llength;j++)
  //     warns[i][j]=' ';
  //   warns[i][llength] ='\0';
  // }

  /* Read in the records */
  // nrecords=0;

  // for (i=0;i<nread;i++) {
  //   getaline(fileunit,cjunk);
  //   for (j=0;j<ncomm;j++) {
  //     ic=indexch(cjunk,comment[j]);
  //     if (ic>0) {
  //       for (k=ic-1;k<llength;k++) cjunk[k]=' ';
  //       cjunk[llength-1]='\0';
  //     }
  //   }
  //   if (len_trim(cjunk)>0) {
  //     adjustl(cjunk,llist[nrecords]);
  //     nrecords++;
  //   }
  //   if (feof(fileunit)) break;
  // }

  /* Now read in the tokens from llist */
  // for (i=0;i<nrecords;i++)
  //   for (j=0;j<llength;j++) {
  //     for (k=0;k<llength-1;k++)
  //       tlist[i][j][k]=' ';
  //     tlist[i][j][llength-1]='\0';
  //   }

  // for (i=0;i<nrecords;i++) {
  //   strcpy(ctemp,llist[i]);
  //   nt=0;
  //   while (len_trim(ctemp)>0) {
  //     ic=len_trim(ctemp)+1;
  //     for (itemp=0;itemp<ndiv;itemp++) {
  //       itemp2=indexch(ctemp,divide[itemp]);
  //       if (itemp2==0) itemp2=len_trim(ctemp)+1;
  //       if (itemp2<ic) ic=itemp2;
  //     }
  //     if (ic>1) {
  //       for (itemp=0;itemp<ic-1;itemp++)
  //         cjunk[itemp]=ctemp[itemp];
  //       cjunk[ic-1]='\0';
  //       adjustl(cjunk,tlist[i][nt]);        
  //       nt++;
  //     }
  //     for (itemp=ic;itemp<strlen(ctemp)+1;itemp++)
  //       cjunk[itemp-ic]=ctemp[itemp];
  //     cjunk[strlen(ctemp)-ic+1]='\0';
  //     adjustl(cjunk,ctemp);        
  //   }
  // }

  /* Check if any of the "labels" in the input file are unrecognized */
  inblock=0; /* false */

  // for (i=0;i<nrecords;i++) {
  //   /* Check if we are in a block */
  //   strcpy(ctemp,tlist[i][0]);
  //   if (strcmp(esdf_reduce(strlwr(ctemp)),"begin")==0) {
  //     inblock=1; /* true */

  //     /* Check if block label is recognized */
  //     strcpy(ctemp,tlist[i][1]); esdf_reduce(strlwr(ctemp));
  //     k=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) k++;
  //     if (k==0) {
  //       strcpy(ctemp,"Label \"");
  //       strcat(ctemp,esdf_reduce(tlist[i][1]));
  //       strcat(ctemp,"\" not in keyword list");
  //       if (countw(ctemp,warns,nwarns)==0) esdf_warn(ctemp);
  //     }

  //     /* Check if "label" is multiply defined in the input file */
  //     ndef=0;
  //     for (j=0;j<nrecords;j++)
  //       if (strcmp(esdf_reduce(tlist[i][1]),
  //             esdf_reduce(tlist[j][1]))==0) ndef++;
  //     strcpy(ctemp,"Label \"");
  //     strcat(ctemp,esdf_reduce(tlist[i][1]));
  //     strcat(ctemp,"\" is multiply defined in the input file. ");

  //     if ((ndef>2)&&(countw(ctemp,warns,nwarns)==0)) esdf_warn(ctemp);
  //   }

    /* Check it is in the list of keywords */
    // strcpy(ctemp,tlist[i][0]); esdf_reduce(strlwr(ctemp));
    // k=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) k++;
    // if ((k==0)&&(!inblock)) {
    //   strcpy(ctemp,"Label \"");
    //   strcat(ctemp,esdf_reduce(tlist[i][0]));
    //   strcat(ctemp,"\" not in keyword list");
    //   if (countw(ctemp,warns,nwarns)==0) esdf_warn(ctemp);
    // }

    // /* Check if "label" is multiply defined in the input file */
    // if (!inblock) {
    //   ndef=0;
    //   for (j=0;j<nrecords;j++)
    //     if (strcmp(esdf_reduce(tlist[i][0]),
    //           esdf_reduce(tlist[j][0]))==0) ndef++;
    //   strcpy(ctemp,"Label \"");
    //   strcat(ctemp,esdf_reduce(tlist[i][0]));
    //   strcat(ctemp,"\" is multiply defined in the input file. ");
    //   if ((ndef>1)&&(countw(ctemp,warns,nwarns)==0)) esdf_warn(ctemp);
    // }
    // }
}

/*   --------------  esdf_string  ----------------------  */
void esdf_string(const char *labl, const char *def, char *out) {
  /* Return the string attached to the "label" */
  int i,kw_number;
  int ind;
  char *pout;
  char label[llength];
  char strT[] = "T";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strT,&kw_number);

  /* Set to default */
  strcpy(out,def);
  pout=out;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      ind=indexstr(llist[i],trim(tlist[i][1]))-1;
      while (llist[i][ind]!='\0') {
        *pout=llist[i][ind];
        pout++; ind++;
      }
      kw_index[kw_number]=i+1;
      break;
    }
  }
  if (pout!=out) *pout='\0';
}
std::string yaml_string(const std::string labl, const std::string def) {
  if (!conf[labl].IsDefined())
      return def;
  else
      return conf[labl].as<std::string>();
}

/*   --------------  esdf_integer  ----------------------  */
int esdf_integer(const char *labl,int def) {
  /* Return the integer value attached to the "label" */
  int i;
  char ctemp[llength];
  int kw_number=0;
  int out;
  char label[llength];
  char strI[] = "I";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strI,&kw_number);

  /* Set to default */
  out=def;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=atoi(tlist[i][1]);
      if ((out==0)&&(atoi(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_integer");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i+2;
      break;
    }
  }

  return out;
}
// -------------- yaml_integer ------------
int yaml_integer(const std::string labl, int def) {
    if (!conf[labl].IsDefined()) {
        return def;
    }
    else
        return conf[labl].as<int>();
}

/*   --------------  esdf_single  ----------------------  */
float esdf_single(const char *labl,float def) {
  /* Return the single precisioned value attached to the "label" */
  float out;
  int i;
  char ctemp[llength];
  int kw_number;
  char label[llength];
  char strS[] = "S";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strS,&kw_number);

  /* Set to default */
  out=def;
  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_single");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i + 1;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_double  ----------------------  */
double esdf_double(const char *labl,double def) {
  /* Return the double precisioned value attached to the "label" */
  int i;
  char ctemp[llength];
  int kw_number;     
  double out;
  char label[llength];
  char strD[] = "D";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label,strD,&kw_number);

  /* Set to default */
  out=def;
  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_double");
        esdf_die(ctemp);
        continue;
      }
      kw_index[kw_number]=i+1;
      break;
    }
  }

  return out;
}
// yaml double 

double yaml_double(const std::string labl, double def) {
  if (!conf[labl].IsDefined()) {
      return def;
  }
  else
      return conf[labl].as<double>();
    
}
/*   --------------  esdf_physical  ----------------------  */
double esdf_physical(const char *labl,double def,char *dunit) {
  /* Return the double precisioned physical value attached to the "label"
     units converted to "dunit"
   */
  int i,j;
  char ctemp[llength],iunit[llength];
  int kw_number;
  double out;
  char label[llength];
  char strP[] = "P";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strP,&kw_number);

  /* Set to default */
  out=def;

  for (i=0;i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=atof(tlist[i][1]);
      if ((out==0)&&(atof(strcat(strcpy(ctemp,"1"),tlist[i][1])))!=10) {
        strcpy(ctemp,"Unable to parse \"");
        strcat(ctemp,esdf_reduce(label));
        strcat(ctemp,"\" in esdf_physical");
        esdf_die(ctemp);
        continue;
      }
      strcpy(iunit,dunit);
      for (j=0;j<llength-strlen(dunit)-1;j++) strcat(iunit," ");
      if (len_trim(tlist[i][2])!=0)
        strcat(iunit,tlist[i][2]);
      out=esdf_convfac(iunit,dunit) * out;
      kw_index[kw_number]=i + 1;
    }
  }
  return out;
}

/*   --------------  esdf_defined  ----------------------  */
bool esdf_defined(const char *labl) {
  /* Is the "label" defined in the input file */
  int i;
  int kw_number;
  bool out;
  char label[llength];
  char strE[] = "E";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strE,&kw_number);

  /* Set to default */
  out=0; /* false */

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=1; /* true */
      kw_index[kw_number]=i+1;
      break;
    }
  }

  return out;
}

/*   --------------  esdf_boolean  ----------------------  */
bool esdf_boolean(const char *labl, bool def) {
  /* Is the "label" defined in the input file */
  int i;
  char positive[3][llength],negative[3][llength];
  char ctemp[llength];
  int kw_number;
  bool out;
  char label[llength];
  char strL[] = "L";

  strcpy(label,labl);
  strcpy(positive[0],"yes");
  strcpy(positive[1],"true");
  strcpy(positive[2],"t");
  strcpy(negative[0],"no");
  strcpy(negative[1],"false");
  strcpy(negative[2],"f");

  /* Check "label" is defined */
  esdf_lablchk(label, strL,&kw_number);

  /* Set to default */
  out=def;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if (strcmp(esdf_reduce(tlist[i][0]),esdf_reduce(label))==0) {
      out=1; /* true */
      kw_index[kw_number]=i+2;
      if (len_trim(tlist[i][1])==0) {
        out=1; /* true */
        break;
      }
      if ((indexstr(positive[0],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(positive[1],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(positive[2],esdf_reduce(tlist[i][1]))>0)) {
        out=1; /* true */
        break;
      }
      if ((indexstr(negative[0],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(negative[1],esdf_reduce(tlist[i][1]))>0) ||
          (indexstr(negative[2],esdf_reduce(tlist[i][1]))>0)) {
        out=0; /* false */
        break;
      }
      strcpy(ctemp, "Unable to parse boolean value");
      esdf_die(ctemp);
    }
  }

  return out;
}

/*   --------------  esdf_block  ----------------------  */
bool esdf_block(const char *labl,int *nlines) {
  int i;
  char ctemp[llength];
  int kw_number;
  bool out;
  char label[llength];
  char strB[] = "B";

  strcpy(label,labl);
  /* Check "label" is defined */
  esdf_lablchk(label, strB,&kw_number);
  strcpy(ctemp,"Block \"");
  strcat(ctemp,esdf_reduce(label));
  strcat(ctemp,"\" not closed correctly");

  out=0; /* false */
  (*nlines)=0;

  for (i=kw_index[kw_number];i<nrecords;i++) {
    /* Search in the first token for "label"
       the first instance is returned */
    if ((strcmp(esdf_reduce(tlist[i][0]),"begin")==0) &&
        (strcmp(esdf_reduce(tlist[i][1]),esdf_reduce(label))==0)) {
      out=1; /* true */
      kw_index[kw_number]=i+1;
      while (strcmp(esdf_reduce(tlist[i+(*nlines)+1][0]),"end")!=0) {
        (*nlines)++;
        if ((*nlines)+i>nrecords) esdf_die(ctemp);
        strcpy(block_data[(*nlines)-1],llist[i+(*nlines)]);
      }
      if (strcmp(esdf_reduce(tlist[i+(*nlines)+1][1]),
            esdf_reduce(label))!=0)
        esdf_die(ctemp);
      break;
    }
  }

  return out;
}

/*   --------------  esdf_reduce  ----------------------  */
char *esdf_reduce(char *in) {
  /* Reduce the string to lower case and remove punctuation */
  const int npunct=2;
  char *end;

  /* Define the punctuation to be removed */
  char punct[npunct];

  punct[0]='.'; punct[1]='-';
  if (in) {
    while (((*in)==' ')||((*in)=='\t')||
        ((*in)==punct[0])||((*in)==punct[1])) in++;
    if (*in) {
      end=in+strlen(in);
      while ((end[-1]==' ')||(end[-1]=='\t')||
          (end[-1]==punct[0])||(end[-1]==punct[1])) end--;
      if (end<in+strlen(in)) (*end)='\0';
    }
  }

  return in;
}

/*   --------------  esdf_convfac  ----------------------  */
double esdf_convfac(char *from,char *to) {
  /* Find the conversion factor between physical units */
  int i,ifrom,ito;
  char ctemp[llength];
  double out;

  /* Find the index numbers of the from and to units */
  ifrom=-1; ito=-1;
  for (i=0;i<nphys;i++) {
    if (strcmp(esdf_reduce(from),esdf_reduce(phy_n[i]))==0) ifrom=i;
    if (strcmp(esdf_reduce(to),esdf_reduce(phy_n[i]))==0) ito=i;
  }

  /* Check that the units were recognized */
  if (ifrom==-1) {
    strcpy(ctemp,"Units not recognized in input file :");
    strcat(ctemp,esdf_reduce(from));
    esdf_die(ctemp);
  }

  if (ito==-1) {
    strcpy(ctemp,"Units not recognized in Program :");
    strcat(ctemp,esdf_reduce(to));
    esdf_die(ctemp);
  }

  /* Check that from and to are of the same dimensions */
  if (phy_d[ifrom]!=phy_d[ito]) {
    strcpy(ctemp,"Dimensions do not match :");
    strcat(ctemp,esdf_reduce(from));
    strcat(ctemp," vs ");
    strcat(ctemp,esdf_reduce(to));
    esdf_die(ctemp);
  }

  /* Set the conversion factor */
  out=phy_u[ifrom]/phy_u[ito];

  return out;
}

/*   --------------  esdf_file  ----------------------  */
void esdf_file(int *unit,char *filename,int *ierr) {
  /* Open an old file */
  (*ierr)=0;

  if ((fileunit=fopen(trim(filename),"r"))==NULL){
    ErrorHandling( "Input file cannot be open" );
    (*ierr)=1;
  }
}

/*   --------------  esdf_lablchk  ----------------------  */
void esdf_lablchk(char *str,char *typ,int *index) {
  /* Check if label is recognized */
  char ctemp[llength];
  char tp[2];
  int i,j;

  strcpy(ctemp,str); esdf_reduce(strlwr(ctemp));
  i=0; for (j=0;j<numkw;j++) if (strcmp(ctemp,kw_label[j])==0) i++;
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" not recognized in keyword list");
  if (i==0) esdf_die(ctemp);
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" is multiply defined");
  if (i>1) esdf_die(ctemp);
  strcpy(ctemp,"Label \"");
  strcat(ctemp,esdf_reduce(str));
  strcat(ctemp,"\" has been used with the wrong type");

  strcpy(tp," ");
  i=0;
  while (strcmp(tp," ")==0) {
    strcpy(ctemp,str);
    if (strcmp(esdf_reduce(strlwr(ctemp)),kw_label[i])==0)
      strncpy(tp,kw_typ[i],1);
    i++;
  }

  (*index)=i-1;
  if (strcmp(typ,tp)!=0) esdf_die(ctemp);
}

/*   --------------  esdf_die  ----------------------  */
void esdf_die(char *str) {
  /* Stop execution due to an error cause by esdf */
  char error[llength]="ESDF ERROR: ";

  printf("%s", strcat(error,trim(str)));    
  printf("\nStopping now\n");

  exit(0);
}

/*   --------------  esdf_warn  ----------------------  */
void esdf_warn(char *str) {
  /* Warning due to an error cause by esdf */

  strcpy(warns[nwarns],str);
  nwarns++;
}

/*   --------------  esdf_close  ----------------------  */
void esdf_close() {
  /* Deallocate the data arrays --- call this before re-initializing */
  int i,j;

  for (i=0;i<nrecords;i++) {
    for (j=0;j<llength;j++)
      free(tlist[i][j]);
    free(tlist[i]);
    free(warns[i]);
    free(llist[i]);
    free(block_data[i]);
  }
  free(tlist);
  free(warns);
  free(llist);
  free(block_data);
}


/************************************************************ 
 * Utilities
 ************************************************************/

/* ------------ GETALINE ---------- */
void getaline(FILE *fp, char *aline) {
  /*
   *  Purpose
   *  =======
   *
   *  Read a line from a file.
   *
   *  Arguments
   *  =========
   *
   *  fp        (input) FILE *
   *            handle of the file
   *  aline     (output) char *
   *            output buffer
   *
   */

  char ch='\0';
  int i=0;

  while (!feof(fp)) {
    ch=fgetc(fp);
    if ((ch=='\n')||(ch=='\r'))
      break;
    aline[i++]=ch;
  }

  if (aline[i-1]==(char)EOF) aline[i-1]='\0';
  else aline[i]='\0';
}

/* ------------ GETLINES ---------- */
void getlines(FILE *fp, char **buffer) {
  /*
   *  Purpose
   *  =======
   *
   *  Load a file to memory.
   *
   *  Arguments
   *  =========
   *
   *  fp        (input) FILE *
   *            handle of the file
   *  buffer    (output) char *
   *            output buffer
   *
   */

  int i=0;

  while (!feof(fp))
    getaline(fp,buffer[i++]);
}

/* ------------ TRIM --------------- */
char *trim(char *in) {
  /*
   *  Purpose
   *  =======
   *
   *  Delete blank characters on both ends of a string.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  in        (input/output) char *
   *            pointer to the original string, changed after the call
   *
   *  Return Value
   *  ============
   *
   *  char *
   *  pointer to the trimmed string
   *
   */

  char *end;

  if (in) {
    while (*in==' '||*in=='\t') in++;
    if (*in) {
      end=in+strlen(in);
      while (end[-1]==' '||end[-1]=='\t') end--;
      (*end)='\0';
    }
  }

  return in;
}

/* ------------ ADJUSTL ------------ */
void adjustl(char *in,char *out) {
  /*
   *  Purpose
   *  =======
   *
   *  Move blank characters from the beginning of the string to the end.
   *
   *  Arguments
   *  =========
   *
   *  in        (input) char *
   *            pointer to the original string
   *  out       (output) char *
   *            pointer to the new string
   *
   */

  char *pin,*pout;
  int i;

  for (i=0;in[i]==' '||in[i]=='\t';i++);
  for (pin=in+i,pout=out;(*pout=*pin);pin++,pout++);
  for (;i>0;i--,pout++)
    *pout=' ';
  *pout='\0';
}

/* ------------ LEN_TRIM ----------- */
int len_trim(char *in) {
  /*
   *  Purpose
   *  =======
   *
   *  Trim a string and calculate its length.
   *  Delete blank characters on both ends of a string.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  in        (input/output) char *
   *            pointer to the original string, changed after the call
   *
   *  Return Value
   *  ============
   *
   *  int
   *  length of the trimmed string
   *
   */

  return strlen(trim(in));
}

/* ------------ INDEX -------------- */
int indexstr(char *string, char *substring) {
  /*
   *  Purpose
   *  =======
   *
   *  Find the first occurence of a substring.
   *
   *  Arguments
   *  =========
   *
   *  string    (input) char *
   *            pointer to the string
   *  substring (input) char *
   *            pointer to the substring
   *
   *  Return Value
   *  ============
   *
   *  int
   *  >0 index of the substring (1 based indexing)
   *  <0 the substring is not found
   *
   */

  char *p,*q;

  p=string;
  q=strstr(string,substring);

  return q-p+1;
}

/* ------------ INDEXCH ------------ */
int indexch(char *str, char ch) {
  /*
   *  Purpose
   *  =======
   *
   *  Find the first occurence of a character.
   *
   *  Arguments
   *  =========
   *
   *  str       (input) char *
   *            pointer to the string
   *  ch        (input) char
   *            the character to be found
   *
   *  Return Value
   *  ============
   *
   *  int
   *  >0 index of the character (1 based indexing)
   *  =0 the character is not found
   *
   */

  char *p,*q;

  p=str;
  q=strchr(str,ch);

  if (q-p+1>0)
    return q-p+1;
  else
    return 0;
}

/* ------------ COUNTW -------------- */
int countw(char *str, char **pool, int nrecords) {
  /*
   *  Purpose
   *  =======
   *
   *  Count the time of occurences of a string.
   *
   *  Arguments
   *  =========
   *
   *  str       (input) char *
   *            the string to be counted
   *  pool      (input) char *[]
   *            the whole string list
   *  nrecords  (input) int
   *            number of strings in the list
   *
   *  Return Value
   *  ============
   *
   *  int
   *  time of occurences of the string
   *
   */

  int i,n=0;

  for (i=0;i<nrecords;i++)
    if (strcmp(str,pool[i])==0)
      n++;

  return n;
}

/* ------------ STRLWR -------------- */
char *strlwr(char *str) {
  /*
   *  Purpose
   *  =======
   *
   *  Convert a string to lower case, if possible.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  str       (input/output) char *
   *            pointer to the original string, keep unchanged
   *
   *  Return Value
   *  ============
   * 
   *  char *
   *  pointer to the new string
   *
   */

  char *p;

  for (p=str;*p;p++)
    if (isupper(*p))
      *p=tolower(*p);

  return str;
}

/* ------------ STRUPR -------------- */
char *strupr(char *str) {
  /*
   *  Purpose
   *  =======
   *
   *  Convert a string of the first letter to upper case,
   *     others to lower case, if possible.
   *  Overwrite the original one.
   *
   *  Arguments
   *  =========
   *
   *  str       (input/output) char *
   *            pointer to the original string, keep unchanged
   *
   *  Return Value
   *  ============
   * 
   *  char *
   *  pointer to the new string
   *
   */

  char *p;

  p=str;
  if (islower(*p)) *p=toupper(*p);
  p++;
  for (;*p;p++)
    if (isupper(*p))
      *p=tolower(*p);

  return str;
}

// *********************************************************************
// Input interface
// *********************************************************************

void ESDFReadInput( const std::string filename ){
  ESDFReadInput( filename.c_str() );
  return ;
}

void
ESDFReadInput ( const char* filename )
{
  Int  mpirank;  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank );
  Int  mpisize;  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );
  Int  nlines;
  const Int MAX_CHAR = 128;
  char  strtmp[MAX_CHAR];
  Real timeSta, timeEnd;

  GetTime( timeSta);
  // Read and distribute the input file
  // if( io_rank == 0 )  
  esdf_init( filename );
  GetTime( timeEnd );
  statusOFS << "Time for reading the raw input file is " <<
      timeEnd - timeSta << " [s]" << std::endl << std::endl;

  // Now each processor can read parameters independently

  // Domain
  {
    Domain& dm = esdfParam.domain;

    if (!conf["Super_Cell"].IsDefined()){
      ErrorHandling("Super_Cell cannot be found.");
    }
    else {
      dm.length[0] = conf["Super_Cell"][0].as<double>();
      dm.length[1] = conf["Super_Cell"][1].as<double>();
      dm.length[2] = conf["Super_Cell"][2].as<double>();
    }

    dm.posStart = Point3( 0.0, 0.0, 0.0 );
  }

  // UPF_File
  {
    Domain& dm = esdfParam.domain;
    if (!conf["UPF_File"].IsDefined()) {
      statusOFS << "UPF_File is not defined. " << std::endl;
      statusOFS << "Use the old format of pseudopotential by defininig PeriodTable in the input file." << std::endl;
      statusOFS << "This option will become deprecated in the future" << std::endl;
      esdfParam.periodTableFile = yaml_string("PeriodTable", "");
      if(esdfParam.periodTableFile.empty())
          ErrorHandling("UPF_File or PeriodTable must be defined.");
    }
    else {
      nlines = 0;
      auto value = conf["UPF_File"];
      for (YAML::const_iterator it = value.begin(); it != value.end(); ++it) {
        nlines++;
      }
      esdfParam.pspFile.resize(nlines);
      int i = 0;
      for (YAML::const_iterator it = value.begin(); it != value.end(); ++it, ++i) {
        esdfParam.pspFile[i] = it->as<std::string>();
      }
    }
  }

  // Atoms
  {
    std::vector<Atom>&  atomList = esdfParam.atomList;
    atomList.clear();

    esdfParam.numAtomType = yaml_integer("Atom_Types_Num", 0);
    if( esdfParam.numAtomType == 0 ){
      ErrorHandling("Atom_Types_Num cannot be found.");
    }

    Int atomType[esdfParam.numAtomType];
    Int atomNum[esdfParam.numAtomType];
    if (conf["Atom_Type"].IsDefined() && conf["Atom_Num"].IsDefined()) {
      for( Int iAtomtype = 0; iAtomtype < esdfParam.numAtomType; iAtomtype++){
          atomType[iAtomtype] = conf["Atom_Type"][iAtomtype].as<int>();
          atomNum[iAtomtype] = conf["Atom_Num"][iAtomtype].as<int>();
      } 
    }
    int atomBegin = 0;
    int atomEnd   = atomNum[0];
    for( Int ityp = 0; ityp < esdfParam.numAtomType; ityp++ ){
      Int type = atomType[ityp];
      // TODO Add supported type list
      if( type == 0 ){
        ErrorHandling( "Atom_Type cannot be found.");
      }
      // FIXME IMPORTANT. The "mass" parameter is removed from the
      // reading list.  Mass can be obtained later with periodtable
      // structure and the atom type.  NOTE that the mass in PeriodTable 
      // is in atomic mass unit (amu), but the mass in
      // atomvec is in atomic unit (au).

      //Int  numAtom;

#if 0 // for yaml atom read

      if( esdf_block("Atom_Bohr", &numAtom ) ){
        // Cartesian coordinate (in the unit of Bohr) 
        Point3 pos;
        Domain& dm = esdfParam.domain;

        for( Int j = 0; j < numAtom; j++ ){
          sscanf(block_data[j],"%lf %lf %lf", 
              &pos[0], &pos[1], &pos[2]);

#if 0
          // Force atom to be in unit cell : assume cuboidal unit cell for this
          while(pos[0] > dm.length[0])
            pos[0] -= dm.length[0];
          while(pos[0] < 0.0)
            pos[0] += dm.length[0];

          while(pos[1] > dm.length[1])
            pos[1] -= dm.length[1];
          while(pos[1] < 0.0)
            pos[1] += dm.length[1];

          while(pos[2] > dm.length[2])
            pos[2] -= dm.length[2];
          while(pos[2] < 0.0)
            pos[2] += dm.length[2];
#endif
          // Force atom to be centered around (0,0,0)
          pos[0] -= IRound(pos[0]/dm.length[0])*dm.length[0];
          pos[1] -= IRound(pos[1]/dm.length[1])*dm.length[1];
          pos[2] -= IRound(pos[2]/dm.length[2])*dm.length[2];

          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );
        }
      }
      else if( esdf_block("Atom_Ang", &numAtom ) ){
        // Cartesian coordinate (in the unit of angstrom) 
        Point3 pos;
        Domain& dm = esdfParam.domain;

        for( Int j = 0; j < numAtom; j++ ){
          sscanf(block_data[j],"%lf %lf %lf", 
              &pos[0], &pos[1], &pos[2]);

#if 0
          // Force atom to be in unit cell : assume cuboidal unit cell for this
          const Real ANG2AU = 1.8897261;
          while(pos[0] > (dm.length[0] / ANG2AU) )
            pos[0] -= (dm.length[0] / ANG2AU);
          while(pos[0] < 0.0)
            pos[0] += (dm.length[0] / ANG2AU);

          while(pos[1] > (dm.length[1] / ANG2AU) )
            pos[1] -= (dm.length[1] / ANG2AU);
          while(pos[1] < 0.0)
            pos[1] += (dm.length[1] / ANG2AU);

          while(pos[2] > (dm.length[2] / ANG2AU) )
            pos[2] -= (dm.length[2] / ANG2AU);
          while(pos[2] < 0.0)
            pos[2] += (dm.length[2] / ANG2AU);
          atomList.push_back( 
              Atom( type, ANG2AU*pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );
#endif
          // Force atom to be centered around (0,0,0)
          pos /= au2ang;
          pos[0] -= IRound(pos[0]/dm.length[0])*dm.length[0];
          pos[1] -= IRound(pos[1]/dm.length[1])*dm.length[1];
          pos[2] -= IRound(pos[2]/dm.length[2])*dm.length[2];
          
          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );

        }
      }
      else if ( esdf_block("Atom_Red", &numAtom) )
#endif //for yaml atom read
      {
        // Reduce coordinate (in the unit of Super_Cell)
        Point3 pos;
        Point3 length = esdfParam.domain.length;
        Domain& dm = esdfParam.domain;
        auto value = conf["Atom_Red"];
        if( ityp > 0){
            atomBegin += atomNum[ityp-1];
            atomEnd += atomNum[ityp];
        }
        YAML::const_iterator it = value.begin();
        YAML::const_iterator it2 = value.begin();
        std::advance(it, atomBegin);
        std::advance(it2, atomEnd);
        for (it; it != it2; ++it) {
          std::vector<double> val = it->as<std::vector<double>>();
          pos[0] = val[0];
          pos[1] = val[1];
          pos[2] = val[2];
          // Force atom to be centered around (0,0,0)
          pos[0] *= dm.length[0];
          pos[1] *= dm.length[1];
          pos[2] *= dm.length[2];

          pos[0] -= IRound(pos[0]/dm.length[0])*dm.length[0];
          pos[1] -= IRound(pos[1]/dm.length[1])*dm.length[1];
          pos[2] -= IRound(pos[2]/dm.length[2])*dm.length[2];

          atomList.push_back( 
              Atom( type, pos, Point3(0.0,0.0,0.0), Point3(0.0,0.0,0.0) ) );
        }
      }
    } // for(ityp)
  }

  // System parameters
  {
    esdfParam.mixMaxDim       = yaml_integer("Mixing_MaxDim", 8);
    esdfParam.mixType         = yaml_string("Mixing_Type", "broyden");
    if( esdfParam.mixType != "anderson" &&
        esdfParam.mixType != "kerker+anderson" &&
        esdfParam.mixType != "pulay" &&
        esdfParam.mixType != "broyden" ){
      ErrorHandling("Invalid DFT mixing type.");
    }

    esdfParam.DGHybridmixMaxDim       = yaml_integer("DGHybrid_Mixing_MaxDim", 8);
    esdfParam.DGHybridmixType         = yaml_string("DGHybrid_Mixing_Type", "broyden");
    if( esdfParam.DGHybridmixType != "anderson" &&
        esdfParam.DGHybridmixType != "kerker+anderson" &&
        esdfParam.DGHybridmixType != "pulay" &&
        esdfParam.DGHybridmixType != "broyden" ){
      ErrorHandling("Invalid DGDFT mixing type.");
    }

    esdfParam.mixVariable     = yaml_string("Mixing_Variable", "density");
    if( esdfParam.mixVariable != "density" &&
        esdfParam.mixVariable != "potential" ){
      ErrorHandling("Invalid PWDFT mixing variable.");
    }

    esdfParam.InnermixVariable   = yaml_string("Inner_Mixing_Variable", "density");
    if( esdfParam.InnermixVariable != "density" &&
        esdfParam.InnermixVariable != "potential" &&
        esdfParam.InnermixVariable != "hamiltonian" &&
        esdfParam.InnermixVariable != "densitymatrix" ){
      ErrorHandling("Invalid DG DFT mixing variable.");
    }

    esdfParam.HybridInnermixVariable   = yaml_string("Hybrid_Inner_Mixing_Variable", "densitymatrix");
    if( esdfParam.HybridInnermixVariable != "hamiltonian" &&
        esdfParam.HybridInnermixVariable != "densitymatrix" ){
      ErrorHandling("Invalid DG Hybrid DFT mixing variable.");
    }

    esdfParam.OutermixVariable   = yaml_string("Outer_Mixing_Variable", "potential");
    if( esdfParam.OutermixVariable != "density" &&
        esdfParam.OutermixVariable != "potential" ){
      ErrorHandling("Invalid Outer mixing variable.");
    }

    // DFT and Hybrid DFT SCF parameters  xmqin
    esdfParam.mixStepLength        = yaml_double( "Mixing_StepLength", 0.7 );
    esdfParam.DGHybridmixStepLength  = yaml_double( "DGHybrid_Mixing_StepLength", 0.2 );
    esdfParam.scfInnerTolerance    = yaml_double( "SCF_Inner_Tolerance", 1e-8 );
    esdfParam.scfInnerMinIter      = yaml_integer( "SCF_Inner_MinIter",  5 );
    esdfParam.scfInnerMaxIter      = yaml_integer( "SCF_Inner_MaxIter",  10 );
    esdfParam.scfOuterTolerance    = yaml_double( "SCF_Outer_Tolerance", 1e-8 );
    esdfParam.scfOuterEnergyTolerance    = yaml_double( "SCF_Outer_Energy_Tolerance", 1e-6 );
    esdfParam.scfOuterMinIter      = yaml_integer( "SCF_Outer_MinIter",   3 );
    esdfParam.scfOuterMaxIter      = yaml_integer( "SCF_Outer_MaxIter",   30 );
    esdfParam.scfPhiMaxIter        = yaml_integer( "SCF_Phi_MaxIter",   20 );
    esdfParam.scfPhiTolerance      = yaml_double( "SCF_Phi_Tolerance",  1e-8 );
    esdfParam.DFTscfInnerTolerance    = yaml_double( "DFT_SCF_Inner_Tolerance", 1e-8 );
    esdfParam.DFTscfInnerMinIter      = yaml_integer( "DFT_SCF_Inner_MinIter",   3 );
    esdfParam.DFTscfInnerMaxIter      = yaml_integer( "DFT_SCF_Inner_MaxIter",   10 );
    esdfParam.DFTscfOuterTolerance    = yaml_double( "DFT_SCF_Outer_Tolerance", 1e-8 );
    esdfParam.DFTscfOuterEnergyTolerance    = yaml_double( "DFT_SCF_Outer_Energy_Tolerance", 1e-6 );
    esdfParam.DFTscfOuterMinIter      = yaml_integer( "DFT_SCF_Outer_MinIter",   3 );
    esdfParam.DFTscfOuterMaxIter      = yaml_integer( "DFT_SCF_Outer_MaxIter",   30 );
    esdfParam.HybridscfInnerTolerance    = yaml_double( "Hybrid_SCF_Inner_Tolerance", 1e-6 );
    esdfParam.HybridscfInnerMinIter      = yaml_integer( "Hybrid_SCF_Inner_MinIter",   3 );
    esdfParam.HybridscfInnerMaxIter      = yaml_integer( "Hybrid_SCF_Inner_MaxIter",   30 );

    esdfParam.scfInnerDMTolerance    = yaml_double( "SCF_Inner_DM_Tolerance", 1e-6 );
    esdfParam.scfInnerHamTolerance    = yaml_double( "SCF_Inner_Ham_Tolerance", 1e-6 );

    esdfParam.HybridscfOuterTolerance    = yaml_double( "Hybrid_SCF_Outer_Tolerance", 1e-6 );
    esdfParam.HybridscfOuterEnergyTolerance    = yaml_double( "Hybrid_SCF_Outer_Energy_Tolerance", 1e-6 );
    esdfParam.HybridscfOuterMinIter      = yaml_integer( "Hybrid_SCF_Outer_MinIter",   3 );
    esdfParam.HybridscfOuterMaxIter      = yaml_integer( "Hybrid_SCF_Outer_MaxIter",   30 );
    esdfParam.HybridPhiMaxIter        = yaml_integer( "Hybrid_SCF_Phi_MaxIter",  20 );
    esdfParam.HybridPhiTolerance      = yaml_double( "Hybrid_SCF_Phi_Tolerance", 1e-6 );

    esdfParam.DGHFXNestedLoop         = yaml_integer("DG_Hybrid_Nested_Loop", "1");
    esdfParam.DGHFXEnergyTolerance    = yaml_double( "DG_HFX_Energy_Tolerance", 1e-6 );

    // TDDFT
    {
      esdfParam.isTDDFT            = yaml_integer( "TDDFT",   0); 
      esdfParam.restartTDDFTStep   = yaml_integer( "Restart_TDDFT_Step", 0 );
      esdfParam.TDDFTautoSaveSteps = yaml_integer( "TDDFT_AUTO_SAVE_STEP", 20);
      esdfParam.isTDDFTEhrenfest   = yaml_integer( "TDDFT_EHRENFEST", 1); 
      esdfParam.isTDDFTVext        = yaml_integer( "TDDFT_VEXT",   1); 
      esdfParam.isTDDFTDipole      = yaml_integer( "TDDFT_DIPOLE",   1); 
      esdfParam.TDDFTVextPolx      = yaml_double( "TDDFT_VEXT_POLX", 1.0);
      esdfParam.TDDFTVextPoly      = yaml_double( "TDDFT_VEXT_POLY", 0.0);
      esdfParam.TDDFTVextPolz      = yaml_double( "TDDFT_VEXT_POLZ", 0.0);
      esdfParam.TDDFTVextFreq      = yaml_double( "TDDFT_VEXT_FREQ", 18.0/27.211385);
      esdfParam.TDDFTVextPhase     = yaml_double( "TDDFT_VEXT_PHASE",0.0);
      esdfParam.TDDFTVextAmp       = yaml_double( "TDDFT_VEXT_AMP",  0.0194);
      esdfParam.TDDFTVextT0        = yaml_double( "TDDFT_VEXT_T0",   13.6056925);
      esdfParam.TDDFTVextTau       = yaml_double( "TDDFT_VEXT_TAU",  13.6056925);

      esdfParam.TDDFTVextEnv       = yaml_string("TDDFT_VEXT_ENV", "gaussian");
      if(esdfParam.TDDFTVextEnv != "gaussian" &&
          esdfParam.TDDFTVextEnv != "constant" &&
          esdfParam.TDDFTVextEnv != "sinsq" &&
          esdfParam.TDDFTVextEnv != "erf" &&
          esdfParam.TDDFTVextEnv != "kick"){
        ErrorHandling("Invalid VEXT Environment .");
      }

      esdfParam.TDDFTMethod        = yaml_string("TDDFT_Method", "PTTRAP");
      if(esdfParam.TDDFTMethod != "PTTRAP" &&
          esdfParam.TDDFTMethod != "RK4"   &&
          esdfParam.TDDFTMethod != "PTTRAPDIIS" ) {
        ErrorHandling("Invalid TDDFT method.");
      }

      esdfParam.TDDFTDeltaT            = yaml_double("TDDFT_DELTA_T",  1.0);
      esdfParam.TDDFTTotalT            = yaml_double("TDDFT_TOTAL_T",  40.0);
      esdfParam.TDDFTKrylovMax         = yaml_integer("TDDFT_KRYLOV_MAX", 30);
      esdfParam.TDDFTKrylovTol         = yaml_double("TDDFT_KRYLOV_TOL",  1.0E-7);
      esdfParam.TDDFTPhiTol            = yaml_double("TDDFT_PHI_TOL",  1.0E-7);
      esdfParam.TDDFTDiisTol           = yaml_double("TDDFT_DIIS_TOL",  1.0E-5);
      esdfParam.TDDFTPhiMaxIter        = yaml_integer("TDDFT_PHI_MAXITER", 20);
      esdfParam.TDDFTDiisMaxIter       = yaml_integer("TDDFT_DIIS_MAXITER", 50);
    }

    esdfParam.hybridMixType         = yaml_string("Hybrid_Mixing_Type", "nested");
    if( esdfParam.hybridMixType != "nested" &&
        esdfParam.hybridMixType != "scdiis" &&
        esdfParam.hybridMixType != "pcdiis" ){
      ErrorHandling("Invalid hybrid mixing type.");
    }

    esdfParam.isHybridACETwicePCDIIS           = yaml_integer( "Hybrid_ACE_Twice_PCDIIS", 1 );

    esdfParam.isHybridACE                      = yaml_integer( "Hybrid_ACE", 1 );
    esdfParam.isHybridActiveInit               = yaml_integer( "Hybrid_Active_Init", 0 );

    esdfParam.isHybridDF                       = yaml_integer( "Hybrid_DF", 0 );
    //esdfParam.numMuHybridDF                    = esdf_double( "Num_Mu_Hybrid_DF", 6.0 );
    //esdfParam.numGaussianRandomHybridDF        = esdf_double( "Num_GaussianRandom_Hybrid_DF", 2.0 );
    //esdfParam.numProcScaLAPACKHybridDF         = esdf_integer( "Num_Proc_ScaLAPACK_Hybrid_DF", mpisize );
    //esdfParam.isHybridDFQRCP                   = esdf_integer( "Hybrid_DF_QRCP", 1 );
    //esdfParam.isHybridDFKmeans                 = esdf_integer( "Hybrid_DF_Kmeans", 0 );


    esdfParam.isDGHFISDF                     = yaml_integer( "DGHF_ISDF", 0 );
    esdfParam.DGHFISDFNumMu                  = yaml_double( "DGHF_ISDF_Num_Mu", 6.0 );

    // esdf_string("Hybrid_DF_Type", "QRCP", strtmp); 
    esdfParam.hybridDFType         = yaml_string("Hybrid_DF_Type", "QRCP");
    if( esdfParam.hybridDFType != "QRCP" &&
        esdfParam.hybridDFType != "Kmeans" &&
        esdfParam.hybridDFType != "Kmeans+QRCP"){
      ErrorHandling("Invalid ISDF type.");
    }

    esdfParam.hybridDFKmeansTolerance          = yaml_double( "Hybrid_DF_Kmeans_Tolerance", 1e-3 );
    esdfParam.hybridDFKmeansMaxIter            = yaml_integer( "Hybrid_DF_Kmeans_MaxIter", 99 );
    esdfParam.hybridDFNumMu                    = yaml_double( "Hybrid_DF_Num_Mu", 6.0 );
    esdfParam.hybridDFNumGaussianRandom        = yaml_double( "Hybrid_DF_Num_GaussianRandom", 2.0 );
    esdfParam.hybridDFNumProcScaLAPACK         = yaml_integer( "Hybrid_DF_Num_Proc_ScaLAPACK", mpisize );
    esdfParam.hybridDFTolerance                = yaml_double( "Hybrid_DF_Tolerance", 1e-20 );
    esdfParam.BlockSizeScaLAPACK               = yaml_integer( "Block_Size_ScaLAPACK", 32 );

    esdfParam.MDscfPhiMaxIter      = yaml_integer( "MD_SCF_Phi_MaxIter", esdfParam.scfPhiMaxIter  );
    esdfParam.MDscfOuterMaxIter    = yaml_integer( "MD_SCF_Outer_MaxIter",  esdfParam.scfOuterMaxIter ); // This is used in DGDFT for energy based SCF

    esdfParam.exxDivergenceType    = yaml_integer( "EXX_Divergence_Type", 1 );

    // Default is no locking
    esdfParam.eigTolerance         = yaml_double( "Eig_Tolerance", 1e-20 );
    esdfParam.eigMinTolerance      = yaml_double( "Eig_Min_Tolerance", 1e-3 );
    esdfParam.eigMinIter           = yaml_integer( "Eig_MinIter",  2 );
    esdfParam.eigMaxIter           = yaml_integer( "Eig_MaxIter",  3 );
    esdfParam.SVDBasisTolerance    = yaml_double( "SVD_Basis_Tolerance", 1e-6 );
    esdfParam.isUseAtomDensity = yaml_integer( "Use_Atom_Density", 0 );
    esdfParam.isUseVLocal      = yaml_integer( "Use_VLocal", 0 );
    esdfParam.isRestartDensity = yaml_integer( "Restart_Density", 0 );
    esdfParam.isRestartWfn     = yaml_integer( "Restart_Wfn", 0 );
    esdfParam.isOutputDensity  = yaml_integer( "Output_Density", 0 );
    esdfParam.isOutputPotential  = yaml_integer( "Output_Potential", 0 );
    esdfParam.isOutputWfn      = yaml_integer( "Output_Wfn", 0 );
    esdfParam.isOutputALBElemLGL      = yaml_integer( "Output_ALB_Elem_LGL", 0 );
    esdfParam.isOutputALBElemUniform  = yaml_integer( "Output_ALB_Elem_Uniform", 0 );
    esdfParam.isOutputWfnExtElem      = yaml_integer( "Output_Wfn_ExtElem", 0 );
    esdfParam.isOutputPotExtElem      = yaml_integer( "Output_Pot_ExtElem", 0 );
    esdfParam.isOutputEigvecCoef      = yaml_integer( "Output_Eigvec_Coef", 0 );
    esdfParam.isCalculateAPosterioriEachSCF = yaml_integer( "Calculate_APosteriori_Each_SCF", 0 );
    esdfParam.isCalculateForceEachSCF       = yaml_integer( "Calculate_Force_Each_SCF", 0 );
    esdfParam.isOutputHMatrix  = yaml_integer( "Output_HMatrix", 0 );

    // Parameters related to Chebyshev Filtering in PWDFT
    {
      esdfParam.First_SCF_PWDFT_ChebyFilterOrder = yaml_integer("First_SCF_PWDFT_ChebyFilterOrder", 40 );
      esdfParam.First_SCF_PWDFT_ChebyCycleNum =  yaml_integer("First_SCF_PWDFT_ChebyCycleNum", 5);
      esdfParam.General_SCF_PWDFT_ChebyFilterOrder = yaml_integer("General_SCF_PWDFT_ChebyFilterOrder", 35);
      esdfParam.PWDFT_Cheby_use_scala = yaml_integer("PWDFT_Cheby_use_scala", 1);
      esdfParam.PWDFT_Cheby_apply_wfn_ecut_filt = yaml_integer("PWDFT_Cheby_use_wfn_ecut_filt",1);
    }

    esdfParam.isHFXFineGrid     = yaml_integer( "Use_HFXFineGrid", 0 );
    esdfParam.ecutWavefunction     = yaml_double( "Ecut_Wavefunction", 40.0 );
    esdfParam.densityGridFactor    = yaml_double( "Density_Grid_Factor", 2.0 );

    // The density grid factor must be an integer
    esdfParam.temperature               = yaml_double( "Temperature", 300.0 );

    esdfParam.Tbeta           = au2K / esdfParam.temperature;

    esdfParam.smearing_scheme = yaml_string("Smearing_Scheme", "FD");

    esdfParam.numExtraState   = yaml_integer( "Extra_States",  0 );
    esdfParam.numUnusedState  = yaml_integer( "Unused_States",  0 );
    esdfParam.isEigToleranceDynamic = yaml_integer( "Eig_Tolerance_Dynamic", 0 );

    esdfParam.pseudoType      = yaml_string("Pseudo_Type", "HGH");
    esdfParam.PWSolver        = yaml_string("PW_Solver", "LOBPCG");
    esdfParam.XCType          = yaml_string("XC_Type", "XC_LDA_XC_TETER93");
    esdfParam.isUseLIBXC      = yaml_integer( "Use_LIBXC",   1);
    esdfParam.VDWType         = yaml_string("VDW_Type", "None");
    esdfParam.numProcScaLAPACKPW  = yaml_integer( "Num_Proc_ScaLAPACK_PW", mpisize );
    esdfParam.scaBlockSize  = yaml_integer( "ScaLAPACK_Block_Size", 32 );
    esdfParam.extraElectron = yaml_integer( "Extra_Electron", 0);

    // PPCG 
    {
      esdfParam.PPCGsbSize = yaml_integer( "PPCG_sbSize", 1);
    }
  }

  // DG
  Index3& numElem = esdfParam.numElem;
  if (conf["Element_Size"].IsDefined()) {
    numElem[0] = conf["Element_Size"][0].as<int>();
    numElem[1] = conf["Element_Size"][1].as<int>();
    numElem[2] = conf["Element_Size"][2].as<int>();
    esdfParam.isDGDFT = true;
  }
  else{
    esdfParam.isDGDFT = false;
    numElem(0) = 1;
    numElem(1) = 1;
    numElem(2) = 1;
  }

  if( esdfParam.isDGDFT ){

    // Instead of grid size, use ecut to determine the number of grid
    // points in the local LGL domain.
    // The LGL grid factor does not need to be an integer.
    esdfParam.LGLGridFactor = yaml_double( "LGL_Grid_Factor", 2.0 );

    esdfParam.GaussInterpFactor = yaml_double( "Gauss_Interp_Factor", 4.0 );

    esdfParam.GaussSigma = yaml_double( "Gauss_Sigma", 0.001 );

    esdfParam.penaltyAlpha  = yaml_double( "Penalty_Alpha", 20.0 );

    // Get the number of basis functions per element
    // NOTE: ALB_Num_Element overwrites the parameter numALB later        
    {
      esdfParam.numALBElem.Resize( numElem[0], numElem[1], numElem[2] );

      Int sizeALBElem;

      Int numALB        = yaml_integer( "ALB_Num", 4 );
      // Use the same number of ALB functions for each element.
      for( Int k = 0; k < numElem[2]; k++ )
        for( Int j = 0; j < numElem[1]; j++ )
          for( Int i = 0; i < numElem[0]; i++ ){
            esdfParam.numALBElem(i,j,k) = numALB;
          }
    }

    // Modification of the potential in the extended element
    {
      // FIXME The potential barrier is now obsolete.
      esdfParam.isPotentialBarrier   = yaml_integer( "Potential_Barrier",  0 );
      esdfParam.potentialBarrierW    = yaml_double( "Potential_Barrier_W", 2.0 );
      esdfParam.potentialBarrierS    = yaml_double( "Potential_Barrier_S", 0.0 );
      esdfParam.potentialBarrierR    = yaml_double( "Potential_Barrier_R", 5.0 );

      // Periodization of the external potential
      esdfParam.isPeriodizePotential = yaml_integer( "Periodize_Potential", 0 );

      esdfParam.distancePeriodize[0] = 0.0;
      esdfParam.distancePeriodize[1] = 0.0;
      esdfParam.distancePeriodize[2] = 0.0;

      if( esdfParam.isPeriodizePotential ){
        if(conf["Distance_Periodize"].IsDefined()){
              esdfParam.distancePeriodize[0] = conf["Distance_Periodize"][0].as<double>();
              esdfParam.distancePeriodize[1] = conf["Distance_Periodize"][1].as<double>();
              esdfParam.distancePeriodize[2] = conf["Distance_Periodize"][2].as<double>();
        }
        else{
          // Default value for DistancePeriodize
          for( Int d = 0; d < DIM; d++ ){
            if( esdfParam.numElem[d] == 1 ){
              esdfParam.distancePeriodize[d] = 0.0;
            }
            else{
              esdfParam.distancePeriodize[d] = 
                esdfParam.domain.length[d] / esdfParam.numElem[d] * 0.5;
            }
          }
        }
      }
    } // Modify the potential

    esdfParam.solutionMethod  = yaml_string("Solution_Method", "diag");
    if( esdfParam.solutionMethod != "diag" &&
        esdfParam.solutionMethod != "pexsi" ){
      ErrorHandling("Invalid solution method for the projected problem.");
    }
    if( esdfParam.solutionMethod == "pexsi" ){
#ifndef _USE_PEXSI_
      ErrorHandling("Usage of PEXSI requires -DPEXSI to be defined in make.inc.");
#endif
    }

    esdfParam.diagSolutionMethod  = yaml_string("Diag_Solution_Method", "scalapack");
    if( esdfParam.diagSolutionMethod != "elpa" &&
        esdfParam.diagSolutionMethod != "scalapack" ){
      ErrorHandling("Invalid Diag solution method for the projected problem.");
    }

    // PEXSI parameters
    esdfParam.numPole           = yaml_integer( "Num_Pole", 60 );
    esdfParam.numProcRowPEXSI   = yaml_integer( "Num_Proc_Row_PEXSI", 1 );
    esdfParam.numProcColPEXSI   = yaml_integer( "Num_Proc_Col_PEXSI", 1 );
    esdfParam.npSymbFact        = yaml_integer( "Num_Proc_Symb_Fact", 
        std::min( 4, esdfParam.numProcRowPEXSI * esdfParam.numProcColPEXSI ) );
    esdfParam.energyGap         = yaml_double( "Energy_Gap", 0.0 );
    esdfParam.spectralRadius    = yaml_double( "Spectral_Radius", 100.0 );
    esdfParam.matrixOrdering    = yaml_integer( "Matrix_Ordering", 0 );
    esdfParam.inertiaCountSteps = yaml_integer( "Inertia_Count_Steps", 10 );
    esdfParam.maxPEXSIIter         = yaml_integer( "Max_PEXSI_Iter", 5 );
    esdfParam.numElectronPEXSITolerance =
      yaml_double( "Num_Electron_PEXSI_Tolerance", 1e-3 );
    esdfParam.muInertiaTolerance =
      yaml_double( "Mu_Inertia_Tolerance", 0.05 );
    esdfParam.muInertiaExpansion =
      yaml_double( "Mu_Inertia_Expansion", 0.3 );
    esdfParam.muPEXSISafeGuard =
      yaml_double( "Mu_PEXSI_SafeGuard", 0.05 );
    esdfParam.muMin             = yaml_double( "Mu_Min", -2.0 );
    esdfParam.muMax             = yaml_double( "Mu_Max", +2.0 );
    esdfParam.pexsiMethod       = yaml_integer( "PEXSI_Method", 1);
    esdfParam.pexsiNpoint       = yaml_integer( "PEXSI_Npoint", 2);

    // Split MPI communicators into row and column communicators

    Domain& dm = esdfParam.domain;

    int dmCol = numElem[0] * numElem[1] * numElem[2];
    int dmRow = mpisize / dmCol;

    if(mpisize == 1){
      dmCol = 1;
      dmRow = 1;
    }

    if( (mpisize % dmCol) != 0 ){
      std::ostringstream msg;
      msg
        << "(mpisize % dmCol) != 0" << std::endl;
      ErrorHandling( msg.str().c_str() );
    }

    dm.comm    = MPI_COMM_WORLD;
    MPI_Comm_split( dm.comm, mpirank / dmRow, mpirank, &dm.rowComm );
    MPI_Comm_split( dm.comm, mpirank % dmRow, mpirank, &dm.colComm );

    MPI_Barrier(dm.rowComm);
    Int mpirankRow;  MPI_Comm_rank(dm.rowComm, &mpirankRow);
    Int mpisizeRow;  MPI_Comm_size(dm.rowComm, &mpisizeRow);

    MPI_Barrier(dm.colComm);
    Int mpirankCol;  MPI_Comm_rank(dm.colComm, &mpirankCol);
    Int mpisizeCol;  MPI_Comm_size(dm.colComm, &mpisizeCol);

    // FFT
    esdfParam.numProcDistFFT  = yaml_integer( "Num_Proc_DistFFT", mpisizeCol );

    // ScaLAPACK parameter
    esdfParam.numProcScaLAPACK    = yaml_integer( "Num_Proc_ScaLAPACK", mpisize );
    if( esdfParam.numProcScaLAPACKPW > mpisizeRow )
      esdfParam.numProcScaLAPACKPW = mpisizeRow;
  } // DG

  // Choose the number of grid points
  // NOTE: This part of the code only applies to DGDFT, since the
  // wavefunction grid and the density grid size is assumed to be a
  // multiple of the number of elements along each dimension.
  //
  // The formula for the number of grid points along each dimension with
  // length L is
  //
  // 1/2 K_max^2 = Ecut,   with K_max = pi N_max / L, 
  //
  // i.e.
  //
  // N_max = \frac{\sqrt{2 E_cut} * L}{pi}.
  //
  // The number of grid point along this dimension is chosen to be the
  // largest even number bigger than N_max.  The number of global grid
  // points is also required to be divisible by the number of elements
  // along that dimension.
  //
  // TODO Later the number of grid points can be improved to only
  // contain the factor of 2, 3 and 5.
  //
  // TODO Current ecutDensity is only used for global grid quantities
  // such as density and potential.  When solving the local problem, the
  // number of grid points for density is still the same as that for the
  // wavefunction.  This constraint can be improved later by saving the
  // wavefunction in the extended element really in the Fourier domain.
  // This real dual grid approach will be done in the next step.
  {
    Domain&  dm       = esdfParam.domain;
    Index3&  numGridWavefunctionElem = esdfParam.numGridWavefunctionElem;
    Index3&  numGridDensityElem = esdfParam.numGridDensityElem;
    Index3&  numGridLGL = esdfParam.numGridLGL;
    Index3   numElem = esdfParam.numElem;

    Point3  elemLength;
    Int npow2;

/*    // Read dense grid numbers for PWDFT if specified in input file
    if (conf["Num_Grid_Fine"].IsDefined()){
      Index3 &numGridFine = dm.numGridFine;
      Index3 &numGrid = dm.numGrid;

      for( Int d = 0; d < DIM; d++ ){
        // numGridFine[d] is always even
        numGridFine[d] = ( conf["Num_Grid_Fine"][d].as<int>() + 1 ) / 2 * 2;
        if( numGridFine[d] < esdfParam.densityGridFactor ){
          ErrorHandling("The number of dense grids is less than densityGridFactor");
        }
        numGrid[d] = std::ceil( numGridFine[d] / esdfParam.densityGridFactor );
      }
    }
*/
    for( Int d = 0; d < DIM; d++ ){
      elemLength[d] = dm.length[d] / numElem[d];
      // the number of grid is assumed to be at least an even number

      // Add implementations for different types of FFT, by xmqin
      esdfParam.FFTtype = yaml_string("FFT_Number_Type", "even");

      if( esdfParam.FFTtype == "even" ){
        numGridWavefunctionElem[d] = 
          std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) * 
            elemLength[d] / PI / 2.0) * 2;
      }
      else if( esdfParam.FFTtype == "odd" ) {
        numGridWavefunctionElem[d] =
          std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) *
            elemLength[d] / PI / 2.0) * 2 + 1;
      }
      else if ( esdfParam.FFTtype == "power" ) {
        npow2 = std::ceil(log(std::ceil(std::sqrt(2.0 * esdfParam.ecutWavefunction) *
              elemLength[d] / PI / 2.0) * 2) /log(2));
        numGridWavefunctionElem[d] = pow(2,npow2);
      }
      else
      {
        ErrorHandling("Invalid type of FFT number");
      }

      numGridDensityElem[d] = std::ceil(numGridWavefunctionElem[d] * esdfParam.densityGridFactor / 2.0) * 2;
      // Set grid number by Ecut if not specified
      if (!conf["Grid_Size"].IsDefined()){
        dm.numGrid[d] = numGridWavefunctionElem[d] * numElem[d];  // Coarse Grid
        dm.numGridFine[d] = numGridDensityElem[d] * numElem[d]; // Fine Frid
      }

      numGridLGL[d] = std::ceil( numGridWavefunctionElem[d] * esdfParam.LGLGridFactor );
    } // for (d)

    if( (!esdfParam.isDGDFT ) && conf["Grid_Size"].IsDefined() ){
      for( Int d = 0; d < DIM; d++ ){
        dm.numGrid[d] = conf["Grid_Size"][d].as<int>();
        dm.numGridFine[d] = std::ceil(dm.numGrid[d] * esdfParam.densityGridFactor / 2.0) * 2;
      }
    }

    if(dm.numGridFine[2] < numElem[0]*numElem[1]*numElem[2] ){
      esdfParam.numProcDistFFT  = yaml_integer( "Num_Proc_DistFFT", dm.numGridFine[2] );
    }
  }

  // Ionic motion
  {
    // Both for geometry optimization and molecular dynamics
    // The default is 0, which means that only static calculation.
    esdfParam.ionMaxIter     = yaml_integer("Ion_Max_Iter", 0);
    esdfParam.ionMove        = yaml_string("Ion_Move", "");

    // Geometry optimization
    esdfParam.geoOptMaxForce = yaml_double( "Geo_Opt_Max_Force", 0.001 );

    // NLCG related parameters
    esdfParam.geoOpt_NLCG_sigma = yaml_double( "Geo_Opt_NLCG_Sigma", 0.02 );

    // FIRE related parameters
    esdfParam.FIRE_Nmin = yaml_integer( "FIRE_Nmin", 5 );		// Compare with LAMMPS
    esdfParam.FIRE_dt = yaml_double("FIRE_Time_Step", 41.3413745758); 	// usually between 0.1-1fs 
    esdfParam.FIRE_atomicmass = yaml_double("FIRE_Atomic_Mass", 4.0); 	// Compare with LAMMPS

    // Molecular dynamics
    Real ionTemperature;
    ionTemperature            = yaml_double( "Ion_Temperature", 300.0 );
    esdfParam.ionTemperature  = ionTemperature;
    esdfParam.TbetaIonTemperature   = au2K / ionTemperature;

    esdfParam.MDTimeStep  = yaml_double("MD_Time_Step", 40.0);
    esdfParam.MDExtrapolationType          = yaml_string("MD_Extrapolation_Type", "linear");
    esdfParam.MDExtrapolationVariable      = yaml_string("MD_Extrapolation_Variable", "density");
    esdfParam.qMass       = yaml_double("Thermostat_Mass", 85000.0);
    esdfParam.langevinDamping       = yaml_double("Langevin_Damping", 0.01);
    esdfParam.kappaXLBOMD           = yaml_double("kappa_XLBOMD", 1.70);
    esdfParam.isRestartPosition     = yaml_integer( "Restart_Position", 0 );
    esdfParam.isRestartVelocity     = yaml_integer( "Restart_Velocity", 0 );
    esdfParam.isOutputPosition      = yaml_integer( "Output_Position", 1 );
    esdfParam.isOutputVelocity      = yaml_integer( "Output_Velocity", 1 );
    esdfParam.isOutputXYZ           = yaml_integer( "Output_XYZ", 1 );

    // Energy based SCF convergence for MD: currently used in DGDFT only
    esdfParam.MDscfEnergyCriteriaEngageIonIter = yaml_integer( "MD_SCF_energy_criteria_engage_ioniter", esdfParam.ionMaxIter + 1); 
    esdfParam.MDscfEtotdiff = yaml_double("MD_SCF_Etot_diff", esdfParam.scfOuterEnergyTolerance);
    esdfParam.MDscfEbanddiff = yaml_double("MD_SCF_Eband_diff", esdfParam.scfOuterEnergyTolerance);
    // Restart position / thermostat
  }

  // Inputs related to Chebyshev Filtered SCF iterations for DG
  {
    // Basic parameters
    esdfParam.Diag_SCFDG_by_Cheby = yaml_integer( "Diag_SCFDG_by_Cheby", 0 );
    esdfParam.SCFDG_Cheby_use_ScaLAPACK = yaml_integer( "SCFDG_Cheby_use_ScaLAPACK", 0 );

    // First SCF step parameters
    esdfParam.First_SCFDG_ChebyFilterOrder = yaml_integer( "First_SCFDG_ChebyFilterOrder", 60 );
    esdfParam.First_SCFDG_ChebyCycleNum = yaml_integer( "First_SCFDG_ChebyCycleNum", 5 );

    // Second stage parameters
    esdfParam.Second_SCFDG_ChebyOuterIter = yaml_integer( "Second_SCFDG_ChebyOuterIter", 3 );
    esdfParam.Second_SCFDG_ChebyFilterOrder = yaml_integer( "Second_SCFDG_ChebyFilterOrder", 60 );
    esdfParam.Second_SCFDG_ChebyCycleNum = yaml_integer( "Second_SCFDG_ChebyCycleNum", 3);

    // General SCF step parameters
    esdfParam.General_SCFDG_ChebyFilterOrder = yaml_integer( "General_SCFDG_ChebyFilterOrder", 60);
    esdfParam.General_SCFDG_ChebyCycleNum = yaml_integer( "General_SCFDG_ChebyCycleNum", 1);
  }

  // Inputs related to Chebyshev polynomial filtered 
  // complementary subspace iteration strategy in DGDFT
  {
    esdfParam.scfdg_use_chefsi_complementary_subspace = yaml_integer("SCFDG_use_CheFSI_complementary_subspace", 0);
    esdfParam.scfdg_chefsi_complementary_subspace_syrk = yaml_integer("SCFDG_CheFSI_complementary_subspace_syrk", 0);
    esdfParam.scfdg_chefsi_complementary_subspace_syr2k = yaml_integer("SCFDG_CheFSI_complementary_subspace_syr2k", esdfParam.scfdg_chefsi_complementary_subspace_syrk);

    esdfParam.scfdg_complementary_subspace_nstates = yaml_integer("SCFDG_complementary_subspace_nstates", int(double(esdfParam.numExtraState)/20.0 + 0.5) );
    esdfParam.scfdg_cs_ioniter_regular_cheby_freq = yaml_integer("SCFDG_CS_ioniter_regular_Cheby_freq", 20 );

    esdfParam.scfdg_cs_bigger_grid_dim_fac = yaml_integer("SCFDG_CS_bigger_grid_dim_fac", 1 );

    // Inner LOBPCG related options
    esdfParam.scfdg_complementary_subspace_lobpcg_iter = yaml_integer("SCFDG_complementary_subspace_inner_LOBPCGiter", 15);
    esdfParam.scfdg_complementary_subspace_lobpcg_tol = yaml_double("SCFDG_complementary_subspace_inner_LOBPCGtol", 1e-8);

    // Inner CheFSI related options
    esdfParam.Hmat_top_states_use_Cheby = yaml_integer("SCFDG_complementary_subspace_use_inner_Cheby", 1);
    esdfParam.Hmat_top_states_ChebyFilterOrder =  yaml_integer("SCFDG_complementary_subspace_inner_Chebyfilterorder", 5);
    esdfParam.Hmat_top_states_ChebyCycleNum = yaml_integer("SCFDG_complementary_subspace_inner_Chebycyclenum", 3);
  }

  // Read position from lastPos.out into esdfParam.atomList[i].pos if isRestartPosition=1
  if(esdfParam.isRestartPosition){
    statusOFS << std::endl 
      << "Read in atomic position from lastPos.out, " << std::endl 
      << "override the atomic positions read from the input file." 
      << std::endl;

    std::vector<Atom>&  atomList = esdfParam.atomList;
    Int numAtom = atomList.size();
    DblNumVec atomposRead(3*numAtom);
    // Only master processor read and then distribute
    if( mpirank == 0 ){
      std::fstream fin;
      fin.open("lastPos.out", std::ios::in);
      if( !fin.good() ){
        ErrorHandling( "Cannot open lastPos.out!" );
      }
      for(Int a=0; a<numAtom; a++){
        fin>> atomposRead[3*a];
        fin>> atomposRead[3*a+1];
        fin>> atomposRead[3*a+2];
      }
      fin.close();
    }
    // Broadcast the atomic position
    MPI_Bcast( atomposRead.Data(), 3*numAtom, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    Point3 pos;
    for(Int a=0; a<numAtom; a++){
      pos = Point3( atomposRead[3*a], atomposRead[3*a+1], atomposRead[3*a+2] );
      atomList[a].pos = pos;
    }
  } // position read in for restart

  // *********************************************************************
  // Some remaining consistency checks
  // *********************************************************************
  if( (esdfParam.pseudoType == "HGH")
      && esdfParam.isUseAtomDensity == true ){
    std::ostringstream msg;
    msg << "For the choice of pseudopotential cannot use atom density as the initial guess";
    ErrorHandling(msg.str());
  }

  // *********************************************************************
  // LRTDDFT
  // *********************************************************************

  
#if 0
{
    esdfParam.isLRTDDFT = yaml_integer("LRTDDFT", 0);
    esdfParam.isLRTDDFTISDF = yaml_integer("LRTDDFT_ISDF", 0);
    esdfParam.isOutputExcitationEnergy = yaml_integer("Output_Excitation_Energy", 0);
    esdfParam.isOutputExcitationWfn = yaml_integer("Output_Excitation_Wfn", 0);
    esdfParam.nvband = yaml_integer("NvBand", 1);
    esdfParam.ncband = yaml_integer("NcBand", 1);
    esdfParam.nkband = yaml_integer("NkBand", 1);
    esdfParam.startband = yaml_integer("OutStartBand", 0);
    esdfParam.endband = yaml_integer("OutEndBand", esdfParam.nkband);
    esdfParam.numMuFacLRTDDFTISDF = yaml_double("NumMuFac_LRTDDFT_ISDF", 1.0);
    esdfParam.numGaussianRandomFacLRTDDFTISDF = yaml_double("NumGaussianRandomFac_LRTDDFT_ISDF", 6.0);
    esdfParam.toleranceLRTDDFT = yaml_double("Tolerance_LRTDDFT", 1e-8);
    esdfParam.maxIterKmeansLRTDDFTISDF = yaml_integer("MaxIterKmeans_LRTDDFT_ISDF", 10);
    esdfParam.toleranceKmeansLRTDDFTISDF = yaml_double("ToleranceKmeans_LRTDDFT_ISDF", 1e-5);

    esdfParam.ipTypeLRTDDFTISDF = yaml_string("IPType_LRTDDFT_ISDF", "QRCP"); 
    esdfParam.eigenSolverLRTDDFT = yaml_string("EigenSolver_LRTDDFT", "LAPACK");
    esdfParam.numProcEigenSolverLRTDDFT  = yaml_integer("NumProcEigenSolver_LRTDDFT", mpisize);
    esdfParam.eigMaxIterLRTDDFT = yaml_integer("EigMaxIter_LRTDDFT", 10);
    esdfParam.eigMinToleranceLRTDDFT = yaml_double("EigMinTolerance_LRTDDFT", 1e-5);
    esdfParam.eigToleranceLRTDDFT = yaml_double("EigTolerance_LRTDDFT", 1e-10);

    // Spectrum
    esdfParam.isOutputExcitationSpectrum = yaml_integer("Spectrum", 0);
    esdfParam.LRTDDFTVextPolx = yaml_double( "LRTDDFT_VEXT_POLX", 1.0);
    esdfParam.LRTDDFTVextPoly = yaml_double( "LRTDDFT_VEXT_POLY", 0.0);
    esdfParam.LRTDDFTVextPolz = yaml_double( "LRTDDFT_VEXT_POLZ", 0.0);
    esdfParam.LRTDDFTOmegagrid = yaml_double( "LRTDDFT_Omega_Grid", 0.1);
    esdfParam.LRTDDFTSigma = yaml_double( "LRTDDFT_Sigma", 0.1);
}
#endif

  return ;
}        // -----  end of function ESDFReadInput  ----- 

void ESDFPrintInput( ){
  int  mpirank;  MPI_Comm_rank( MPI_COMM_WORLD, &mpirank ); 
  int  mpisize;  MPI_Comm_size( MPI_COMM_WORLD, &mpisize );

  // If the product of the number of elements is 1, recognize this as a PWDFT calculation

  PrintBlock(statusOFS, "Common information");

  Print(statusOFS, "Super cell                           = ",  esdfParam.domain.length );
  Print(statusOFS, "Grid Wavefunction                    = ",  esdfParam.domain.numGrid ); 
  Print(statusOFS, "Grid Density                         = ",  esdfParam.domain.numGridFine );
  Print(statusOFS, "Mixing dimension                     = ",  esdfParam.mixMaxDim );
  Print(statusOFS, "Mixing variable                      = ",  esdfParam.mixVariable );
  Print(statusOFS, "Mixing DFT type                      = ",  esdfParam.mixType );
  Print(statusOFS, "Mixing Hybrid DGDFT type             = ",  esdfParam.DGHybridmixType );
  Print(statusOFS, "Mixing Steplength                    = ",  esdfParam.mixStepLength);

  Print(statusOFS, "DGHF Mixing Dimension                = ",  esdfParam.DGHybridmixMaxDim );
  Print(statusOFS, "DGHF Mixing Steplength               = ",  esdfParam.DGHybridmixStepLength );

  Print(statusOFS, "DG InnerSCF Mixing variable          = ",  esdfParam.InnermixVariable );
  Print(statusOFS, "DG OuterSCF Mixing variable          = ",  esdfParam.OutermixVariable );

  Print(statusOFS, "DG DFT OuterSCF Tolerance            = ",  esdfParam.DFTscfOuterTolerance);
  Print(statusOFS, "DG DFT OuterSCF Energy Tolerance     =",   esdfParam.DFTscfOuterEnergyTolerance);
  Print(statusOFS, "DG DFT OuterSCF MinIter              = ",  esdfParam.DFTscfOuterMinIter);
  Print(statusOFS, "DG DFT OuterSCF MaxIter              = ",  esdfParam.DFTscfOuterMaxIter);
  Print(statusOFS, "DG DFT InnerSCF Tolerance            = ",  esdfParam.DFTscfInnerTolerance);
//  Print(statusOFS, "DG DFT InnerSCF Energy Tolerance     =",   esdfParam.DFTscfInnerEnergyTolerance);
  Print(statusOFS, "DG DFT InnerSCF MinIter              = ",  esdfParam.DFTscfInnerMinIter);
  Print(statusOFS, "DG DFT InnerSCF MaxIter              = ",  esdfParam.DFTscfInnerMaxIter);
  Print(statusOFS, "DG Hybrid InnerSCF Tolerance         = ",  esdfParam.HybridscfInnerTolerance);
//  Print(statusOFS, "DG Hybrid InnerSCF Energy Tolerance  =",   esdfParam.HybridscfInnerEnergyTolerance);
  Print(statusOFS, "DG Hybrid InnerSCF MinIter           = ",  esdfParam.HybridscfInnerMinIter);
  Print(statusOFS, "DG Hybrid InnerSCF MaxIter           = ",  esdfParam.HybridscfInnerMaxIter);
  Print(statusOFS, "DG Hybrid OuterSCF MinIter           = ",  esdfParam.HybridscfOuterMinIter);
  Print(statusOFS, "DG Hybrid OuterSCF MaxIter           = ",  esdfParam.HybridscfOuterMaxIter);
  Print(statusOFS, "DG Hybrid OuterSCF Tolerance         = ",  esdfParam.HybridscfOuterTolerance);
  Print(statusOFS, "DG Hybrid OuterSCF Energy Tolerance  =",   esdfParam.HybridscfOuterEnergyTolerance);
  Print(statusOFS, "DG Hybrid ALBs MaxIter               = ",  esdfParam.HybridPhiMaxIter);
  Print(statusOFS, "DG Hybrid ALBs Tolerance             = ",  esdfParam.HybridPhiTolerance);
   Print(statusOFS,"DG Hybrid Nested Loop Type           = ", esdfParam.DGHFXNestedLoop );
  Print(statusOFS, "Temperature                          = ", esdfParam.temperature, "[K]");
  Print(statusOFS, "au2K                                 = ",  au2K );
  Print(statusOFS, "Tbeta                                = ",  esdfParam.Tbeta );
  Print(statusOFS, "Temperature2                          = ",  au2K / esdfParam.Tbeta, "[K]");
  Print(statusOFS, "Extra states                         = ",  esdfParam.numExtraState  );
  Print(statusOFS, "Smearing scheme                      = ",  esdfParam.smearing_scheme );
  Print(statusOFS, "PeriodTable File                     = ",  esdfParam.periodTableFile );
  Print(statusOFS, "Pseudo Type                          = ",  esdfParam.pseudoType );
  Print(statusOFS, "PW Solver                            = ",  esdfParam.PWSolver );
  Print(statusOFS, "XC Type                              = ",  esdfParam.XCType );

  Print(statusOFS, "Hybrid ACE                           = ",  esdfParam.isHybridACE);
  Print(statusOFS, "Hybrid DF                            = ",  esdfParam.isHybridDF);
  Print(statusOFS, "DGHF ISDF                            = ",  esdfParam.isDGHFISDF);
  Print(statusOFS, "DGHF ISDF Num Mu                     = ",  esdfParam.DGHFISDFNumMu);

  Print(statusOFS, "Hybrid Active Init                   = ",  esdfParam.isHybridActiveInit);
  Print(statusOFS, "Hybrid Mixing Type                   = ",  esdfParam.hybridMixType);
  Print(statusOFS, "Hybrid DF Num Mu                     = ",  esdfParam.hybridDFNumMu);
  Print(statusOFS, "Hybrid DF Num GaussianRandom         = ",  esdfParam.hybridDFNumGaussianRandom);
  Print(statusOFS, "Hybrid DF Tolerance                  = ",  esdfParam.hybridDFTolerance);

  Print(statusOFS, "SCF Outer Tol                        = ",  esdfParam.scfOuterTolerance);
  Print(statusOFS, "SCF Outer MaxIter                    = ",  esdfParam.scfOuterMaxIter);
  Print(statusOFS, "SCF Free Energy Per Atom Tol         = ",  esdfParam.scfOuterEnergyTolerance);
  Print(statusOFS, "Eig Min Tolerence                    = ",  esdfParam.eigMinTolerance);
  Print(statusOFS, "Eig Tolerence                        = ",  esdfParam.eigTolerance);
  Print(statusOFS, "Eig MaxIter                          = ",  esdfParam.eigMaxIter);
  Print(statusOFS, "Eig Tolerance Dyn                    = ",  esdfParam.isEigToleranceDynamic);
  Print(statusOFS, "Num unused state                     = ",  esdfParam.numUnusedState);
  Print(statusOFS, "EcutWavefunction                     = ",  esdfParam.ecutWavefunction);
  Print(statusOFS, "Density GridFactor                   = ",  esdfParam.densityGridFactor);

  Print(statusOFS, "Use Atom Density                     = ",  esdfParam.isUseAtomDensity);
  Print(statusOFS, "Use VLocal                           = ",  esdfParam.isUseVLocal);
  Print(statusOFS, "RestartDensity                       = ",  esdfParam.isRestartDensity);
  Print(statusOFS, "RestartWfn                           = ",  esdfParam.isRestartWfn);
  Print(statusOFS, "OutputDensity                        = ",  esdfParam.isOutputDensity);
  Print(statusOFS, "OutputPotential                      = ",  esdfParam.isOutputPotential);
  Print(statusOFS, "Number of Extra Electron             = ",  esdfParam.extraElectron);

  // Ionic motion
  if( esdfParam.ionMove != "" ){
    Print(statusOFS, "");
    Print(statusOFS, "Ion move mode                        = ",  esdfParam.ionMove);
    Print(statusOFS, "Max steps for ion                    = ",  esdfParam.ionMaxIter);
    Print(statusOFS, "MD time step                         = ",  esdfParam.MDTimeStep);
    Print(statusOFS, "Ion Temperature                      = ",  esdfParam.ionTemperature, "[K]");
    Print(statusOFS, "Thermostat mass                      = ",  esdfParam.qMass);
    Print(statusOFS, "Langevin damping                     = ",  esdfParam.langevinDamping);
    Print(statusOFS, "RestartPosition                      = ",  esdfParam.isRestartPosition);
    Print(statusOFS, "RestartVelocity                      = ",  esdfParam.isRestartVelocity);
    Print(statusOFS, "OutputPosition                       = ",  esdfParam.isOutputPosition );
    Print(statusOFS, "OutputVelocity                       = ",  esdfParam.isOutputVelocity   );
    Print(statusOFS, "Output XYZ format                    = ",  esdfParam.isOutputXYZ );
    Print(statusOFS, "Force tol for geoopt                 = ",  esdfParam.geoOptMaxForce );
    Print(statusOFS, "MD extrapolation type                = ",  esdfParam.MDExtrapolationType);
    Print(statusOFS, "MD extrapolation variable            = ",  esdfParam.MDExtrapolationVariable);
    Print(statusOFS, "MD SCF Phi MaxIter                   = ",  esdfParam.MDscfPhiMaxIter);
    Print(statusOFS, "MD SCF Outer MaxIter                 = ",  esdfParam.MDscfOuterMaxIter);
    Print(statusOFS, "MD SCF Energy Criteria Engage Iter   = ",  esdfParam.MDscfEnergyCriteriaEngageIonIter);
    Print(statusOFS, "MD SCF Etot diff                     = ",  esdfParam.MDscfEtotdiff);
    Print(statusOFS, "MD SCF Eband diff                    = ",  esdfParam.MDscfEbanddiff);
    Print(statusOFS, "");
  }

  if(esdfParam.isTDDFT) {
    PrintBlock(statusOFS, "TDDFT information");
    Print(statusOFS, "TDDFT Method                         = ",  esdfParam.TDDFTMethod   );
    Print(statusOFS, "TDDFT Ehrenfest dynamics             = ",  esdfParam.isTDDFTEhrenfest);
    Print(statusOFS, "TDDFT Delta T                        = ",  esdfParam.TDDFTDeltaT   );
    Print(statusOFS, "TDDFT Total T                        = ",  esdfParam.TDDFTTotalT   );
    Print(statusOFS, "TDDFT Restart Step                   = ",  esdfParam.restartTDDFTStep);
    Print(statusOFS, "TDDFT auto save for Restart          = ",  esdfParam.TDDFTautoSaveSteps);
    Print(statusOFS, "TDDFT KRYLOV Iteration Max           = ",  esdfParam.TDDFTKrylovMax);
    Print(statusOFS, "TDDFT KRYLOV Tolerance               = ",  esdfParam.TDDFTKrylovTol);
    Print(statusOFS, "TDDFT V external                     = ",  esdfParam.isTDDFTVext   );
    Print(statusOFS, "TDDFT Calculate Dipole               = ",  esdfParam.isTDDFTDipole );
    Print(statusOFS, "TDDFT Environment                    = ",  esdfParam.TDDFTVextEnv  );
    Print(statusOFS, "TDDFT Polarization X                 = ",  esdfParam.TDDFTVextPolx );
    Print(statusOFS, "TDDFT Polarization Y                 = ",  esdfParam.TDDFTVextPoly );
    Print(statusOFS, "TDDFT Polarization Z                 = ",  esdfParam.TDDFTVextPolz );
    Print(statusOFS, "TDDFT V external Frequencey          = ",  esdfParam.TDDFTVextFreq );
    Print(statusOFS, "TDDFT V external Phase               = ",  esdfParam.TDDFTVextPhase);
    Print(statusOFS, "TDDFT V external Amplitude           = ",  esdfParam.TDDFTVextAmp  );
    Print(statusOFS, "TDDFT V external T0                  = ",  esdfParam.TDDFTVextT0   );
    Print(statusOFS, "TDDFT V external Tau                 = ",  esdfParam.TDDFTVextTau  );
    Print(statusOFS, "TDDFT DIIS Tolerance                 = ",  esdfParam.TDDFTDiisTol  );
    Print(statusOFS, "TDDFT Phi Tolerance                  = ",  esdfParam.TDDFTPhiTol   );
    Print(statusOFS, "TDDFT DIIS MaxIter                   = ",  esdfParam.TDDFTDiisMaxIter);
    Print(statusOFS, "TDDFT Phi MaxIter                    = ",  esdfParam.TDDFTPhiMaxIter);
  }

  if( esdfParam.isDGDFT ){
    PrintBlock(statusOFS, "DGDFT information");
    // FIXME Potentially obsolete potential barriers
    Print(statusOFS, "Penalty Alpha                        = ",  esdfParam.penaltyAlpha );
    Print(statusOFS, "Element size                         = ",  esdfParam.numElem ); 
    Print(statusOFS, "Wfn Elem GridSize                    = ",  esdfParam.numGridWavefunctionElem );
    Print(statusOFS, "Rho Elem GridSize                    = ",  esdfParam.numGridDensityElem ); 
    Print(statusOFS, "LGL Grid size                        = ",  esdfParam.numGridLGL ); 
    Print(statusOFS, "LGL GridFactor                       = ",  esdfParam.LGLGridFactor);

    Print(statusOFS, "SVD Basis Tol                        = ",  esdfParam.SVDBasisTolerance);
    Print(statusOFS, "SCF Inner Tol                        = ",  esdfParam.scfInnerTolerance);
    Print(statusOFS, "SCF Inner MaxIter                    = ",  esdfParam.scfInnerMaxIter);

    Print(statusOFS, "Solution Method   = ",  esdfParam.solutionMethod );
    if( esdfParam.solutionMethod == "diag" ){
      Print(statusOFS, "Number of procs for ScaLAPACK        = ",  esdfParam.numProcScaLAPACK); 
      Print(statusOFS, "Number of procs for ScaLAPACK (PW)   = ",  esdfParam.numProcScaLAPACKPW); 
      Print(statusOFS, "ScaLAPACK block                      = ",  esdfParam.scaBlockSize); 
    }

    if( esdfParam.solutionMethod == "pexsi" ){
      Print(statusOFS, "Number of poles                    = ",  esdfParam.numPole); 
      Print(statusOFS, "Nproc row PEXSI                    = ",  esdfParam.numProcRowPEXSI); 
      Print(statusOFS, "Nproc col PEXSI                    = ",  esdfParam.numProcColPEXSI); 
      Print(statusOFS, "Nproc for symbfact                 = ",  esdfParam.npSymbFact); 
      Print(statusOFS, "Energy gap                         = ",  esdfParam.energyGap); 
      Print(statusOFS, "Spectral radius                    = ",  esdfParam.spectralRadius); 
      Print(statusOFS, "Matrix ordering                    = ",  esdfParam.matrixOrdering); 
      Print(statusOFS, "Inertia before SCF                 = ",  esdfParam.inertiaCountSteps);
      Print(statusOFS, "MuMin0                             = ",  esdfParam.muMin); 
      Print(statusOFS, "MuMax0                             = ",  esdfParam.muMax); 
      Print(statusOFS, "NumElectron tol                    = ",  esdfParam.numElectronPEXSITolerance); 
      Print(statusOFS, "mu Inertia tol                     = ",  esdfParam.muInertiaTolerance); 
      Print(statusOFS, "mu Inertia expand                  = ",  esdfParam.muInertiaExpansion); 
      Print(statusOFS, "Max PEXSI iter (deprecated)        = ",  esdfParam.maxPEXSIIter); 
      Print(statusOFS, "mu PEXSI safeguard (deprecated)    = ",  esdfParam.muPEXSISafeGuard); 
    }
    // TODO Chebyshev

    Print(statusOFS, "OutputALBElemLGL                   = ",  esdfParam.isOutputALBElemLGL);
    Print(statusOFS, "OutputALBElemUniform               = ",  esdfParam.isOutputALBElemUniform);
    Print(statusOFS, "OutputWfnExtElem                   = ",  esdfParam.isOutputWfnExtElem);
    Print(statusOFS, "OutputPotExtElem                   = ",  esdfParam.isOutputPotExtElem);
    Print(statusOFS, "OutputHMatrix                      = ",  esdfParam.isOutputHMatrix );

    Print(statusOFS, "Force each step (deprecated)       = ",  
        esdfParam.isCalculateForceEachSCF );

    // FIXME A posteriori
    Print(statusOFS, "A Posteriori error each step       = ",  
        esdfParam.isCalculateAPosterioriEachSCF);

    if( esdfParam.isPeriodizePotential ){
      Print(statusOFS, "PeriodizePotential                 = ",  esdfParam.isPeriodizePotential);
      Print(statusOFS, "DistancePeriodize                  = ",  esdfParam.distancePeriodize);
    }

    if( esdfParam.isPotentialBarrier ){
      Print(statusOFS, "Potential Barrier = ",  esdfParam.isPotentialBarrier);
      Print(statusOFS, "Barrier W         = ",  esdfParam.potentialBarrierW);
      Print(statusOFS, "Barrier S         = ",  esdfParam.potentialBarrierS);
      Print(statusOFS, "Barrier R         = ",  esdfParam.potentialBarrierR);
    }
  } // DG
  else{
    PrintBlock(statusOFS, "PWDFT information");

    // FIXME For DG as well later
    Print(statusOFS, "SCF Phi MaxIter                      = ",  esdfParam.scfPhiMaxIter);
    Print(statusOFS, "SCF Phi Tol                          = ",  esdfParam.scfPhiTolerance);
    Print(statusOFS, "Hybrid ACE                           = ",  esdfParam.isHybridACE);
    Print(statusOFS, "Hybrid DF                            = ",  esdfParam.isHybridDF);
    Print(statusOFS, "Hybrid Active Init                   = ",  esdfParam.isHybridActiveInit);
    Print(statusOFS, "Hybrid Mixing Type                   = ",  esdfParam.hybridMixType);
    Print(statusOFS, "Hybrid DF Num Mu                     = ",  esdfParam.hybridDFNumMu);
    Print(statusOFS, "Hybrid DF Num GaussianRandom         = ",  esdfParam.hybridDFNumGaussianRandom);
    Print(statusOFS, "Hybrid DF Tolerance                  = ",  esdfParam.hybridDFTolerance);
    Print(statusOFS, "EXX div type                         = ",  esdfParam.exxDivergenceType);

    if( esdfParam.PWSolver == "LOBPCGScaLAPACK" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
      Print(statusOFS, "Number of procs for ScaLAPACK (PW)   = ",  esdfParam.numProcScaLAPACKPW); 
      Print(statusOFS, "ScaLAPACK block                      = ",  esdfParam.scaBlockSize); 
    }

    if( esdfParam.PWSolver == "PPCG" || esdfParam.PWSolver == "PPCGScaLAPACK" ){
      Print(statusOFS, "Subblock size (sbSize) in PPCG       = ",  esdfParam.PPCGsbSize); 
    }

#if 0
    if( esdfParam.isLRTDDFT ){
      PrintBlock(statusOFS, "PWDFT LRTDDFT");

      Print(statusOFS, "LRTDDFT                              = ",  esdfParam.isLRTDDFT);
      Print(statusOFS, "Output_Excitation_Energy             = ",  esdfParam.isOutputExcitationEnergy);
      Print(statusOFS, "Output_Excitation_Wfn                = ",  esdfParam.isOutputExcitationWfn);
      Print(statusOFS, "Nvband                               = ",  esdfParam.nvband);
      Print(statusOFS, "Ncband                               = ",  esdfParam.ncband);
      Print(statusOFS, "ISDF Parameters");

      Print(statusOFS, "MuFac                                = ",  esdfParam.numMuFacLRTDDFTISDF);
      Print(statusOFS, "GaussianRandomFac                    = ",  esdfParam.numGaussianRandomFacLRTDDFTISDF);
      Print(statusOFS, "LRTDDFT_Tolerance                    = ",  esdfParam.toleranceLRTDDFT);
      Print(statusOFS, "hybridDFKmeansMaxIter_ISDF           = ",  esdfParam.maxIterKmeansLRTDDFTISDF);      

      Print(statusOFS, "LOBPCG Parameters");
      Print(statusOFS, "Nkband                               = ", esdfParam.nkband);
      Print(statusOFS, "OutStartband                         = ", esdfParam.startband);
      Print(statusOFS, "OutEndband                           = ", esdfParam.endband);
      Print(statusOFS, "useLessProcessInLOBPCG               = ", esdfParam.numProcEigenSolverLRTDDFT);
      Print(statusOFS, "eigMaxIter_LRTDDFT                   = ", esdfParam.eigMaxIterLRTDDFT);
      Print(statusOFS, "eigMinTolerance_LRTDDFT              = ", esdfParam.eigMinToleranceLRTDDFT);
      Print(statusOFS, "eigTolerance_LRTDDFT                 = ", esdfParam.eigToleranceLRTDDFT);

      Print(statusOFS, "LRTDDFT Spectrum Parameters");
      Print(statusOFS, "LRTDDFT_Spectrum                     = ", esdfParam.isOutputExcitationSpectrum);
      Print(statusOFS, "LRTDDFT_VEXT_POLX                    = ", esdfParam.LRTDDFTVextPolx);
      Print(statusOFS, "LRTDDFT_VEXT_POLY                    = ", esdfParam.LRTDDFTVextPoly);
      Print(statusOFS, "LRTDDFT_VEXT_POLZ                    = ", esdfParam.LRTDDFTVextPolz);
      Print(statusOFS, "LRTDDFT_Omega_Grid                   = ", esdfParam.LRTDDFTOmegagrid);
      Print(statusOFS, "LRTDDFT_Sigma                        = ", esdfParam.LRTDDFTSigma);
    }
#endif

  } // PW

  // Only master processor output information containing all atoms
  //

#if ( _DEBUGlevel_ >= 1 )
  if( mpirank == 0 ){
    const std::vector<Atom>&  atomList = esdfParam.atomList;

    Print(statusOFS, ""); 
    Print(statusOFS, "NumAtom = ", (int)atomList.size()); 
    Print(statusOFS, "Atom Type and Coordinates (possibly after shifting into unit cell)");
    Print(statusOFS, ""); 

    for(Int i=0; i < atomList.size(); i++) {
      Print(statusOFS, "Type = ", atomList[i].type, "Position  = ", atomList[i].pos);
    }
  }
#endif

  Print(statusOFS, ""); 
  return ;
}        // -----  end of function ESDFPrintInput  ----- 

} // namespace esdf
} // namespace dgdft
