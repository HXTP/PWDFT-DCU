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
/// @file tinyvec_impl.hpp
/// @brief Implementation of tiny vectors of dimension 2 and 3.
/// @date 2010-09-20
#ifndef  _TINYVEC_IMPL_HPP_
#define  _TINYVEC_IMPL_HPP_

#include "tinyvec_decl.hpp"

namespace dgdft{

// *********************************************************************
// Tiny vectors of dimension 3.
// *********************************************************************

template <class F> 
  inline F&
  Vec3T<F>::operator() ( Int i ) 
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator()  ----- 

template <class F> 
  inline const F&
  Vec3T<F>::operator() ( Int i ) const
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator()  ----- 

template <class F> 
  inline F&
  Vec3T<F>::operator[] ( Int i ) 
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator[]  ----- 

template <class F> 
  inline const F&
  Vec3T<F>::operator[] ( Int i ) const
  {
    if( i < 0 || i > 2 ){
      ErrorHandling( "Index is out of bound." );
    }
    return v_[i];
  }         // -----  end of method Vec3T::operator[]  ----- 

// *********************************************************************
// Vec3T: Compare
// *********************************************************************
template <class F> inline bool operator==(const Vec3T<F>& a, const Vec3T<F>& b) {
  return (a[0]==b[0] && a[1]==b[1] && a[2]==b[2]);
}
template <class F> inline bool operator!=(const Vec3T<F>& a, const Vec3T<F>& b) {
  return !(a==b);
}
template <class F> inline bool operator> (const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator< (const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator>=(const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return true;
}
template <class F> inline bool operator<=(const Vec3T<F>& a, const Vec3T<F>& b) {
  for(Int i=0; i<3; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return true;
}

// *********************************************************************
// Vec3T: Numerical operations
// *********************************************************************
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = -a[i]; return r;
}
template <class F> inline Vec3T<F> operator+ (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]+b[i]; return r; 
}
template <class F> inline Vec3T<F> operator- (const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]-b[i]; return r;
}
template <class F> inline Vec3T<F> operator* (F scl, const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec3T<F> operator* (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec3T<F> operator/ (const Vec3T<F>& a, F scl) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]/scl;  return r;
}
template <class F> inline F operator* (const Vec3T<F>& a, const Vec3T<F>& b) {
  F sum=F(0); for(Int i=0; i<3; i++) sum=sum+a(i)*b(i); return sum;
}
template <class F> inline F dot       (const Vec3T<F>& a, const Vec3T<F>& b) {
  return a*b;
}
template <class F> inline Vec3T<F> operator^ (const Vec3T<F>& a, const Vec3T<F>& b) {
  return Vec3T<F>(a(1)*b(2)-a(2)*b(1), a(2)*b(0)-a(0)*b(2), a(0)*b(1)-a(1)*b(0));
}
template <class F> inline Vec3T<F> cross     (const Vec3T<F>& a, const Vec3T<F>& b) { 
  return a^b; 
}

// *********************************************************************
// Vec3T: Element wise numerical operations
// *********************************************************************
template <class F> inline Vec3T<F> ewmin(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::min(a[i], b[i]); return r;
}
template <class F> inline Vec3T<F> ewmax(const Vec3T<F>& a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::max(a[i], b[i]); return r;
}
template <class F> inline Vec3T<F> ewabs(const Vec3T<F>& a) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = std::abs(a[i]); return r;
}
template <class F> inline Vec3T<F> ewmul(const Vec3T<F>&a, const Vec3T<F>& b) {
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]*b[i]; return r;
}
template <class F> inline Vec3T<F> ewdiv(const Vec3T<F>&a, const Vec3T<F>& b) { 
  Vec3T<F> r;  for(Int i=0; i<3; i++) r[i] = a[i]/b[i]; return r;
}
template <class F> inline Vec3T<F> ewrnd(const Vec3T<F>&a) { //round
  Vec3T<F> r;  for(Int i=0; i<3; i++)    r[i] = round(a[i]);  return r;
}

// *********************************************************************
// Vec3T: Accumulative boolean operations
// *********************************************************************
template <class F> inline bool allequ(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)==b(i));  return res;
}
template <class F> inline bool allneq(const Vec3T<F>& a, const Vec3T<F>& b) {
  return !(a==b);
}
template <class F> inline bool allgtt(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)> b(i));  return res; 
}
template <class F> inline bool alllst(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)< b(i));  return res; 
}
template <class F> inline bool allgoe(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)    res = res && (a(i)>=b(i));  return res; 
}
template <class F> inline bool allloe(const Vec3T<F>& a, const Vec3T<F>& b) {
  bool res = true;  for(Int i=0; i<3; i++)   res = res && (a(i)<=b(i));  return res; 
}


// *********************************************************************
// Vec3T: Input and output
// *********************************************************************
template <class F> std::istream& operator>>(std::istream& is, Vec3T<F>& a) {
  for(Int i=0; i<3; i++) is>>a[i]; return is;
}
template <class F> std::ostream& operator<<(std::ostream& os, const Vec3T<F>& a) { 
  for(Int i=0; i<3; i++) os<<a[i]<<" "; return os;
}


// *********************************************************************
// Tiny vectors of dimension 2.
// *********************************************************************


// *********************************************************************
// Vec2T: Compare
// *********************************************************************
template <class F> inline bool operator==(const Vec2T<F>& a, const Vec2T<F>& b) {
  return (a[0]==b[0] && a[1]==b[1]);
}
template <class F> inline bool operator!=(const Vec2T<F>& a, const Vec2T<F>& b) {
  return !(a==b);
}
template <class F> inline bool operator> (const Vec2T<F>& a, const Vec2T<F>& b) {
  for(int i=0; i<2; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator< (const Vec2T<F>& a, const Vec2T<F>& b) {
  for(int i=0; i<2; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return false;
}
template <class F> inline bool operator>=(const Vec2T<F>& a, const Vec2T<F>& b) {
  for(int i=0; i<2; i++) {
    if(     a[i]>b[i])      return true;
    else if(a[i]<b[i])      return false;
  }
  return true;
}
template <class F> inline bool operator<=(const Vec2T<F>& a, const Vec2T<F>& b) {
  for(int i=0; i<2; i++) {
    if(     a[i]<b[i])      return true;
    else if(a[i]>b[i])      return false;
  }
  return true;
}

// *********************************************************************
// Vec2T: Numerical operations
// *********************************************************************
template <class F> inline Vec2T<F> operator- (const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = -a[i]; return r;
}
template <class F> inline Vec2T<F> operator+ (const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]+b[i]; return r; 
}
template <class F> inline Vec2T<F> operator- (const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]-b[i]; return r;
}
template <class F> inline Vec2T<F> operator* (F scl, const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec2T<F> operator* (const Vec2T<F>& a, F scl) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = scl*a[i];  return r;
}
template <class F> inline Vec2T<F> operator/ (const Vec2T<F>& a, F scl) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]/scl;  return r;
}
template <class F> inline F operator* (const Vec2T<F>& a, const Vec2T<F>& b) {
  F sum=F(0); for(int i=0; i<2; i++) sum=sum+a(i)*b(i); return sum;
}
template <class F> inline F dot       (const Vec2T<F>& a, const Vec2T<F>& b) {
  return a*b;
}

// *********************************************************************
// Vec2T: Element wise numerical operations
// *********************************************************************
template <class F> inline Vec2T<F> ewmin(const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = std::min(a[i], b[i]); return r;
}
template <class F> inline Vec2T<F> ewmax(const Vec2T<F>& a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = std::max(a[i], b[i]); return r;
}
template <class F> inline Vec2T<F> ewabs(const Vec2T<F>& a) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = std::abs(a[i]); return r;
}
template <class F> inline Vec2T<F> ewmul(const Vec2T<F>&a, const Vec2T<F>& b) {
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]*b[i]; return r;
}
template <class F> inline Vec2T<F> ewdiv(const Vec2T<F>&a, const Vec2T<F>& b) { 
  Vec2T<F> r;  for(int i=0; i<2; i++) r[i] = a[i]/b[i]; return r;
}
template <class F> inline Vec2T<F> ewrnd(const Vec2T<F>&a) { //round
  Vec2T<F> r;  for(int i=0; i<2; i++)    r[i] = round(a[i]);  return r;
}

// *********************************************************************
// Vec2T: Input and output
// *********************************************************************
template <class F> std::istream& operator>>(std::istream& is, Vec2T<F>& a) {
  for(int i=0; i<2; i++) is>>a[i]; return is;
}
template <class F> std::ostream& operator<<(std::ostream& os, const Vec2T<F>& a) { 
  for(int i=0; i<2; i++) os<<a[i]<<" "; return os;
}

} // namespace dgdft

#endif // _TINYVEC_IMPL_HPP_
