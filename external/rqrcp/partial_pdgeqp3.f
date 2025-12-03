C A PARTIAL VERSION OF PDGEQP3
C AUTHOR: JIANWEI XIAO AND JULIEN LANGOU

      SUBROUTINE PARTIAL_PDGEQP3( PARTIAL, M, N, A, DESCA, IPIV, TAU,
     $                    DWORK, LDWORK, IWORK, LIWORK, INFO )
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
      INTEGER            INFO, LDWORK, LIWORK, M, N, PARTIAL
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
        CALL PXERBLA( ICTXT, 'PDGEQP3_PARTIAL', -INFO )
        RETURN
      ELSE IF( LQUERY ) THEN
        RETURN
      END IF
C
C     Quick return if possible.
C
C     MN = MIN( M, N )
      MN = PARTIAL
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
C *** Last line of PARTIAL_PDGEQP3 ***
      END
