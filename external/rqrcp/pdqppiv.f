      SUBROUTINE PDQPPIV( M, N, A, IA, JA, DESCA, IPIV )
*
*  -- ScaLAPACK routine (version 1.7) --
*     University of Tennessee, Knoxville, Oak Ridge National Laboratory,
*     and University of California, Berkeley.
*     May 1, 1997
*
*     .. Scalar Arguments ..
      INTEGER            IA, JA, M, N
*     ..
*     .. Array Arguments ..
      INTEGER            DESCA( * ), IPIV( * )
      DOUBLE PRECISION   A( * )
*     ..
*
*  Purpose
*  =======
*
*  PDQPPIV applies to sub( A ) = A(IA:IA+M-1,JA:JA+N-1) the pivots
*  returned by PDGEQPF in reverse order for checking purposes.
*
*  Notes
*  =====
*
*  Each global data object is described by an associated description
*  vector.  This vector stores the information required to establish
*  the mapping between an object element and its corresponding process
*  and memory location.
*
*  Let A be a generic term for any 2D block cyclicly distributed array.
*  Such a global array has an associated description vector DESCA.
*  In the following comments, the character _ should be read as
*  "of the global array".
*
*  NOTATION        STORED IN      EXPLANATION
*  --------------- -------------- --------------------------------------
*  DTYPE_A(global) DESCA( DTYPE_ )The descriptor type.  In this case,
*                                 DTYPE_A = 1.
*  CTXT_A (global) DESCA( CTXT_ ) The BLACS context handle, indicating
*                                 the BLACS process grid A is distribu-
*                                 ted over. The context itself is glo-
*                                 bal, but the handle (the integer
*                                 value) may vary.
*  M_A    (global) DESCA( M_ )    The number of rows in the global
*                                 array A.
*  N_A    (global) DESCA( N_ )    The number of columns in the global
*                                 array A.
*  MB_A   (global) DESCA( MB_ )   The blocking factor used to distribute
*                                 the rows of the array.
*  NB_A   (global) DESCA( NB_ )   The blocking factor used to distribute
*                                 the columns of the array.
*  RSRC_A (global) DESCA( RSRC_ ) The process row over which the first
*                                 row of the array A is distributed.
*  CSRC_A (global) DESCA( CSRC_ ) The process column over which the
*                                 first column of the array A is
*                                 distributed.
*  LLD_A  (local)  DESCA( LLD_ )  The leading dimension of the local
*                                 array.  LLD_A >= MAX(1,LOCr(M_A)).
*
*  Let K be the number of rows or columns of a distributed matrix,
*  and assume that its process grid has dimension p x q.
*  LOCr( K ) denotes the number of elements of K that a process
*  would receive if K were distributed over the p processes of its
*  process column.
*  Similarly, LOCc( K ) denotes the number of elements of K that a
*  process would receive if K were distributed over the q processes of
*  its process row.
*  The values of LOCr() and LOCc() may be determined via a call to the
*  ScaLAPACK tool function, NUMROC:
*          LOCr( M ) = NUMROC( M, MB_A, MYROW, RSRC_A, NPROW ),
*          LOCc( N ) = NUMROC( N, NB_A, MYCOL, CSRC_A, NPCOL ).
*  An upper bound for these quantities may be computed by:
*          LOCr( M ) <= ceil( ceil(M/MB_A)/NPROW )*MB_A
*          LOCc( N ) <= ceil( ceil(N/NB_A)/NPCOL )*NB_A
*
*  Arguments
*  =========
*
*  M       (global input) INTEGER
*          The number of rows to be operated on, i.e. the number of rows
*          of the distributed submatrix sub( A ). M >= 0.
*
*  N       (global input) INTEGER
*          The number of columns to be operated on, i.e. the number of
*          columns of the distributed submatrix sub( A ). N >= 0.
*
*  A       (local input/local output) DOUBLE PRECISION pointer into the
*          local memory to an array of dimension (LLD_A, LOCc(JA+N-1)).
*          On entry, the local pieces of the M-by-N distributed matrix
*          sub( A ) which is to be permuted. On exit, the local pieces
*          of the distributed permuted submatrix sub( A ) * Inv( P ).
*
*  IA      (global input) INTEGER
*          The row index in the global array A indicating the first
*          row of sub( A ).
*
*  JA      (global input) INTEGER
*          The column index in the global array A indicating the
*          first column of sub( A ).
*
*  DESCA   (global and local input) INTEGER array of dimension DLEN_.
*          The array descriptor for the distributed matrix A.
*
*  IPIV    (local input) INTEGER array, dimension LOCc(JA+N-1).
*          On exit, if IPIV(I) = K, the local i-th column of sub( A )*P
*          was the global K-th column of sub( A ). IPIV is tied to the
*          distributed matrix A.
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DTYPE_,
     $                   LLD_, MB_, M_, NB_, N_, RSRC_
      PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DTYPE_ = 1,
     $                     CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6,
     $                     RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
*     ..
*     .. Local Scalars ..
      INTEGER            IACOL, ICOFFA, ICTXT, IITMP, IPVT, IPCOL,
     $                   IPROW, ITMP, J, JJ, JJA, KK, MYCOL, MYROW,
     $                   NPCOL, NPROW, NQ
*     ..
*     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, IGEBR2D, IGEBS2D, IGERV2D,
     $                   IGESD2D, IGAMN2D, INFOG1L, PDSWAP
*     ..
*     .. External Functions ..
      INTEGER            INDXL2G, NUMROC
      EXTERNAL           INDXL2G, NUMROC
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MIN, MOD
*     ..
*     .. Executable Statements ..
*
*     Get grid parameters
*
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      CALL INFOG1L( JA, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ), JJA,
     $              IACOL )
      ICOFFA = MOD( JA-1, DESCA( NB_ ) )
      NQ = NUMROC( N+ICOFFA, DESCA( NB_ ), MYCOL, IACOL, NPCOL )
      IF( MYCOL.EQ.IACOL )
     $   NQ = NQ - ICOFFA
*
      DO 20 J = JA, JA+N-2
*
         IPVT = JA+N-1
         ITMP = JA+N
*
*        Find first the local minimum candidate for pivoting
*
         CALL INFOG1L( J, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ),
     $                 JJ, IACOL )
         DO 10 KK = JJ, JJA+NQ-1
            IF( IPIV( KK ).LT.IPVT )THEN
               IITMP = KK
               IPVT = IPIV( KK )
            END IF
   10    CONTINUE
*
*        Find the global minimum pivot
*
         CALL IGAMN2D( ICTXT, 'Rowwise', ' ', 1, 1, IPVT, 1, IPROW,
     $                 IPCOL, 1, -1, MYCOL )
*
*        Broadcast the corresponding index to the other process columns
*
         IF( MYCOL.EQ.IPCOL ) THEN
            ITMP = INDXL2G( IITMP, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ),
     $                      NPCOL )
            CALL IGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, ITMP, 1 )
            IF( IPCOL.NE.IACOL ) THEN
               CALL IGERV2D( ICTXT, 1, 1, IPIV( IITMP ), 1, MYROW,
     $                       IACOL )
            ELSE
               IF( MYCOL.EQ.IACOL )
     $            IPIV( IITMP ) = IPIV( JJ )
            END IF
         ELSE
            CALL IGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, ITMP, 1, MYROW,
     $                    IPCOL )
            IF( MYCOL.EQ.IACOL .AND. IPCOL.NE.IACOL )
     $         CALL IGESD2D( ICTXT, 1, 1, IPIV( JJ ), 1, MYROW, IPCOL )
         END IF
*
*        Swap the columns of A
*
         CALL PDSWAP( M, A, IA, ITMP, DESCA, 1, A, IA, J, DESCA, 1 )
*
   20 CONTINUE
*
*     End of PDQPPIV
*
      END
