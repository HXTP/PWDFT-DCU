      SUBROUTINE PDGEQP3( M, N, A, DESCA, IPIV, TAU, DWORK, LDWORK,
     $                    IWORK, LIWORK, INFO )
C
C
C     RELEASE $Revision: 1.31 $. COPYRIGHT 2001 (See CONTRIBUTOR).
C
C     PURPOSE
C
C     To compute a QR factorization with column pivoting
C                            A * P = Q * R.
C     using block Householder transformations
C
C     ARGUMENTS
C
C     Input/Output Parameters
C
C     M       (global input) INTEGER
C             The number of rows to be operated on, i.e. the number of
C             rows of the distributed matrix A. M >= 0.
C
C     N       (global input) INTEGER
C             The number of columns to be operated on, i.e. the number
C             of columns of the distributed matrix A. N >= 0.
C
C     A       (local input/local output) DOUBLE PRECISION pointer into
C             the local memory to an array, dimension (LLD_A,LOCc(N))
C             On entry, the local pieces of the M-by-N distributed
C             matrix A which is to be factored.
C             On exit, the elements on and above the diagonal of A
C             contain the min(M,N) by N upper trapezoidal matrix R (R
C             is upper triangular if M >= N); the elements below the
C             diagonal, with the array TAU, represent the orthogonal
C             matrix Q as a product of elementary reflectors.
C
C     DESCA   (global and local input) INTEGER array, dimension DLEN_
C             The array descriptor for the distributed matrix A.
C
C     IPIV    (local output) INTEGER array, dimension LOCc(N)
C             On exit, if IPIV(I) = K, the local i-th column of A*P
C             was the global K-th column of A. IPIV is tied to the
C             distributed matrix A.
C
C     TAU     (local output) DOUBLE PRECISION, array, dimension
C             LOCc(MIN(M,N)). This array must contain the scalar factors
C             TAU of the elementary reflectors. TAU is tied to the
C             distributed matrix A.
C
C     Workspace
C
C     DWORK   (local workspace) DOUBLE PRECISION array,
C             dimension (LDWORK)
C             On exit, DWORK(1) returns the minimal and optimal LDWORK.
C
C     LDWORK  (local input) INTEGER
C             The dimension of the real array used for workspace.
C             LDWORK >= LOCp(NB)*LOCq(N) + 2 * LOCq(N) +
C                       max( NB, LOCp(M), LOCq(N) ).
C
C             If LDWORK = -1, then LDWORK is global input and a
C             workspace query is assumed; the routine only calculates
C             the minimum and optimal size for all work arrays. Each of
C             these values is returned in the first entry of the
C             corresponding work array, and no error message is issued
C             by PXERBLA.
C
C     IWORK   (local workspace) INTEGER array, dimension (LIWORK)
C             On exit, IWORK(1) returns the minimal and optimal LIWORK.
C
C     LIWORK  (local input) INTEGER
C             The dimension of the integer array used for workspace.
C             LIWORK >= LOCq(N).
C
C             If LIWORK = -1, then LIWORK is global input and a
C             workspace query is assumed; the routine only calculates
C             the minimum and optimal size for all work arrays. Each
C             of these values is returned in the first entry of the
C             corresponding work array, and no error message is issued
C             by PXERBLA.
C
C     Error Indicator
C
C     INFO    (global output) INTEGER
C             = 0: Successful exit.
C             < 0: If the i-th argument is an array and the j-entry had
C                  an illegal value, then INFO = -(i*100+j), if the
C                  i-th argument is a scalar and had an illegal value,
C                  then INFO = -i.
C             = K, 1<=K<=MAXIT: The K-th matrix in sequence for A was
C                               singular.
C             = MAXIT+1: The sequence did not converge.
C
C     METHOD
C
C     The method simply delays the application of transformations
C     and constructs a block Householder transformation that is
C     then applied.
C
C     REFERENCES
C
C     [1] Quintana-Orti, G., Sun, X. and Bischof, C.
C         A BLAS-3 version of the QR factorization with column
C         pivoting
C         SIAM J. Sci. Computing, 19(5), pp. 1486-1494, 1998.
C
C     NUMERICAL ASPECTS
C
C     The factorization computed is numerically equivalent to that
C     obtained in the tradional QR factorization with column pivoting
C     and the computational cost is of the same order.
C
C     CONTRIBUTOR
C
C     P. Benner
C       Faculty of Mathematics, Chemnitz University of Technology
C       D-09107 Chemnitz, Germany
C       benner@mathematik.tu-chemnitz.de
C     E. S. Quintana-Orti, G. Quintana-Orti
C       Depto. de Ingenieria y Ciencia de Computadores, Univ. Jaume I
C       12.080 Castellon, Spain
C       {quintana,gquintan}@icc.uji.es
C     Based on $Revision: 1.31 $ of Parallel Library for Control (PLIC)
C     $Date: 2010/03/24 13:13:28 $, http://www.hpca.uji.es/
C
C     PARALLEL EXECUTION RECOMMENDATIONS
C
C     Restrictions:
C     o The distribution blocks must be square (MB=NB).
C
C     REVISIONS
C
C     None.
C
C     KEYWORDS
C
C     QR factorization with column pivoting, block Householder
C     transformations.
C
C     ******************************************************************
C
      IMPLICIT NONE
C
C     .. Parameters ..
      INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DTYPE_,
     $                   LLD_, MB_, M_, NB_, N_, RSRC_
      PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DTYPE_ = 1,
     $                     CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6,
     $                     RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
C     ..
C     .. Scalar Arguments ..
      INTEGER            INFO, LDWORK, LIWORK, M, N
C     ..
C     .. Array Arguments ..
      INTEGER            DESCA( * ), IPIV( * ), IWORK( * )
      DOUBLE PRECISION   A( * ), DWORK( * ), TAU( * )
C     ..
C     .. Local Scalars ..
      LOGICAL            LQUERY
      INTEGER            ICTXT, IPG, IPN1, IPN2, IRWK, J, JACOL, JJA,
     $                   LIWMIN, LWMIN, MN, MYCOL, MYROW, NB, NBIN,
     $                   NBOUT, NPA, NPCOL, NPG, NPROW, NQA, NQG, OFFSET
C     ..
C     .. Local Arrays ..
      INTEGER            DESCG( DLEN_ ), DESCN( DLEN_ ), IDUM( 1 )
C     ..
C     .. External Functions ..
      INTEGER            ICEIL, INDXG2P, NUMROC
      EXTERNAL           ICEIL, INDXG2P, NUMROC
C     ..
C     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, CHK1MAT, DESCSET, INFOG1L,
     $                   PCHK1MAT, PDGEQP3S, PDNRM2, PXERBLA
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC          DBLE, MAX, MIN
C     ..
C     .. Executable Statements ..
C
C     Get grid parameters.
C
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      NB = DESCA( NB_ )
C
C     Test the input parameters.
C
      INFO = 0
      IF( NPROW.EQ.-1 ) THEN
        INFO = -(400+CTXT_)
      ELSE
        CALL CHK1MAT( M, 1, N, 2, 1, 1, DESCA, 4, INFO )
C
        IF( INFO.EQ.0 ) THEN
          NPG = NUMROC( NB, DESCA( MB_ ), MYROW, DESCA( RSRC_ ), NPROW )
          NQG = NUMROC(  N, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
          NPA = NUMROC( M, DESCA( MB_ ), MYROW, DESCA( RSRC_ ), NPROW )
          NQA = NUMROC( N, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
          LWMIN = NPG * NQG + 2 * NQA + MAX( NB, NPA, NQA )
          LIWMIN = NQA
C
          DWORK( 1 ) = DBLE( LWMIN )
          IWORK( 1 ) = LIWMIN
          LQUERY = ( ( LDWORK.EQ.-1 ).OR.( LIWORK.EQ.-1 ) )
          IF( DESCA( MB_ ).NE.DESCA( NB_ ) ) THEN
            INFO = -(400+NB_)
          ELSE IF( LDWORK.LT.LWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -8
          ELSE IF( LIWORK.LT.LIWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -10
          END IF
        END IF
C
        CALL PCHK1MAT( M, 1, N, 2, 1, 1, DESCA, 4, 0, IDUM, IDUM,
     $                 INFO )
      END IF
C
      IF( INFO.NE.0 ) THEN
        CALL PXERBLA( ICTXT, 'PDGEQP3', -INFO )
        RETURN
      ELSE IF( LQUERY ) THEN
        RETURN
      END IF
C
C     Quick return if possible.
C
      MN = MIN( M, N )
      IF( MN.EQ.0 )
     $  RETURN
C
C     Create and initialize auxiliary array descriptors.
C
      NPG = NUMROC( NB, NB, MYROW, DESCA( RSRC_ ), NPROW )
      NQG = NUMROC( N,  NB, MYCOL, DESCA( CSRC_ ), NPCOL )
      CALL DESCSET( DESCG, NB, N, NB, NB, DESCA( RSRC_ ),
     $              DESCA( CSRC_ ), ICTXT, MAX( 1, NPG ) )
      CALL DESCSET( DESCN, 1, N, 1, NB, MYROW, DESCA( CSRC_ ), ICTXT,
     $              1 )
      NPA = NUMROC( M, DESCA( MB_ ), MYROW, DESCA( RSRC_ ), NPROW )
      NQA = NUMROC( N, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
C
C     Set workspace pointers.
C
      IPN1 = 1
      IPN2 = IPN1 + NQA
      IPG  = IPN2 + NQA
      IRWK = IPG  + NPG*NQG
C
C     Initialize the array of pivots and compute the column norms.
C     ============================================================
C
      DO J = 1, N
C
        CALL INFOG1L( J, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ),
     $                JJA, JACOL )
        IF( MYCOL.EQ.JACOL ) THEN
          IPIV( JJA ) = J
C
          CALL PDNRM2( M, DWORK( IPN1-1+JJA ), A, 1, J, DESCA, 1 )
          DWORK( IPN2-1+JJA ) = DWORK( IPN1-1+JJA )
        END IF
C
      END DO
C
C     *************************
C     * Compute factorization *
C     *************************
C
      J = 1
C
C     Loop for processing blocks.
C
 1000 IF( J.LE.MN ) THEN
C
        NBIN = MIN( NB, MN-J+1 )
        OFFSET = J-1
        CALL PDGEQP3S( M, N, OFFSET, A, DESCA, DWORK( IPG ), DESCG,
     $                 DWORK( IPN1 ), DWORK( IPN2 ), DESCN, IPIV, TAU,
     $                 NBIN, NBOUT, DWORK( IRWK ), IWORK )
        J = J + NBOUT
C
        GO TO 1000
C
      END IF
C
      DWORK( 1 ) = DBLE( LWMIN )
      IWORK( 1 ) = LIWMIN
C
      RETURN
C
C *** Last line of PDGEQP3 ***
      END
