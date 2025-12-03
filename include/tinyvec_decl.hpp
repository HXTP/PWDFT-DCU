/*
   Copyright (c) 2012 The Regents of the University of California,
   through Lawrence Berkeley National Laboratory.  

Authors: Lexing Ying and Lin Lin

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
/// @file tinyvec_decl.hpp
/// @brief Tiny vectors of dimension 2 and 3.
/// @date 2010-09-20
#ifndef  _TINYVEC_DECL_HPP_
#define  _TINYVEC_DECL_HPP_

#include "environment.hpp"

namespace dgdft{

// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F> class Vec3T {
private:
  F v_[3];
public:
  enum{ X=0, Y=1, Z=2 };
  //------------CONSTRUCTOR AND DESTRUCTOR 
  Vec3T()              { v_[0]=F(0);    v_[1]=F(0);    v_[2]=F(0); }
  Vec3T(const F* f)    { v_[0]=f[0];    v_[1]=f[1];    v_[2]=f[2]; }
  Vec3T(const F a, const F b, const F c)   { v_[0]=a;       v_[1]=b;       v_[2]=c; }
  Vec3T(const Vec3T& c){ v_[0]=c.v_[0]; v_[1]=c.v_[1]; v_[2]=c.v_[2]; }
  ~Vec3T() {}
  //------------POINTER and ACCESS
  operator F*()             { return &v_[0]; }
  operator const F*() const { return &v_[0]; }
  F* Data()                 { return &v_[0]; }  //access array
  F& operator()(Int i);
  const F& operator()(Int i) const;
  F& operator[](Int i);
  const F& operator[](Int i) const;
  //------------ASSIGN
  Vec3T& operator= ( const Vec3T& c ) { v_[0] =c.v_[0]; v_[1] =c.v_[1]; v_[2] =c.v_[2]; return *this; }
  Vec3T& operator+=( const Vec3T& c ) { v_[0]+=c.v_[0]; v_[1]+=c.v_[1]; v_[2]+=c.v_[2]; return *this; }
  Vec3T& operator-=( const Vec3T& c ) { v_[0]-=c.v_[0]; v_[1]-=c.v_[1]; v_[2]-=c.v_[2]; return *this; }
  Vec3T& operator*=( const F& s )     { v_[0]*=s;       v_[1]*=s;       v_[2]*=s;       return *this; }
  Vec3T& operator/=( const F& s )     { v_[0]/=s;       v_[1]/=s;       v_[2]/=s;       return *this; }
  //-----------LENGTH
  F l1( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]); return sum; }
  F linfty( void ) const  { F cur=F(0); for(Int i=0; i<3; i++) cur=std::max(cur,std::abs(v_[i])); return cur; }
  F l2( void )     const  { F sum=F(0); for(Int i=0; i<3; i++) sum=sum+std::abs(v_[i]*v_[i]); return std::sqrt(sum); }
  F prod( void )   const  { return v_[0]*v_[1]*v_[2]; }
};

// Commonly used Vec3T types

typedef Vec3T<Real>   Point3;
typedef Vec3T<Int>    Index3;
namespace { Point3 _trigger_template_instantiation_point3; }
// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F>
class Vec2T {
private:
  F v_[2];
public:
  enum{ X=0, Y=1 };
  //------------CONSTRUCTOR AND DESTRUCTOR 
  Vec2T()              { v_[0]=F(0);    v_[1]=F(0); }  //Vec2T(F f)           { v_[0]=f;       v_[1]=f; }
Vec2T(const F* f)    { v_[0]=f[0];    v_[1]=f[1]; }
Vec2T(F a,F b)       { v_[0]=a;       v_[1]=b; }
Vec2T(const Vec2T& c){ v_[0]=c.v_[0]; v_[1]=c.v_[1]; }
~Vec2T() {}
//------------POINTER and ACCESS
operator F*()             { return &v_[0]; }
operator const F*() const { return &v_[0]; }
F* data()                { return &v_[0]; }  //access array
F& operator()(int i)             { assert(i<2); return v_[i]; }
const F& operator()(int i) const { assert(i<2); return v_[i]; }
F& operator[](int i)             { assert(i<2); return v_[i]; }
const F& operator[](int i) const { assert(i<2); return v_[i]; }
//------------ASSIGN
Vec2T& operator= ( const Vec2T& c ) { v_[0] =c.v_[0]; v_[1] =c.v_[1]; return *this; }
Vec2T& operator+=( const Vec2T& c ) { v_[0]+=c.v_[0]; v_[1]+=c.v_[1]; return *this; }
Vec2T& operator-=( const Vec2T& c ) { v_[0]-=c.v_[0]; v_[1]-=c.v_[1]; return *this; }
Vec2T& operator*=( const F& s )     { v_[0]*=s;       v_[1]*=s;       return *this; }
Vec2T& operator/=( const F& s )     { v_[0]/=s;       v_[1]/=s;       return *this; }
//-----------LENGTH...
F l1( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+std::abs(v_[i]); return sum; }
F linfty( void ) const  { F cur=F(0); for(int i=0; i<2; i++) cur=std::max(cur,std::abs(v_[i])); return cur; }
F l2( void )     const  { F sum=F(0); for(int i=0; i<2; i++) sum=sum+std::abs(v_[i]*v_[i]); return std::sqrt(sum); }
F prod( void )   const  { return v_[0]*v_[1]; }
};

// Commonly used Vec2T types
typedef Vec2T<double> Point2;
typedef Vec2T<int>    Index2;

} // namespace dgdft

#endif // _TINYVEC_DECL_HPP_
