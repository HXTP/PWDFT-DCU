! A PARTIAL VERSION OF PDGEQRF
! AUTHOR: JIANWEI XIAO

      SUBROUTINE PARTIAL_PDGEQRF( M, N, K, A, IA, JA, DESCA, TAU, WORK,
     $                    LWORK, INFO )
*     .. Scalar Arguments ..
      INTEGER            IA, INFO, JA, LWORK, M, N, K
*     ..
*     .. Array Arguments ..
      INTEGER            DESCA( * )
      DOUBLE PRECISION   A( * ), TAU( * ), WORK( * )
*     .. Parameters ..
      INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DTYPE_,
     $                   LLD_, MB_, M_, NB_, N_, RSRC_
      PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DTYPE_ = 1,
     $                     CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6,
     $                     RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
*     ..
*     .. Local Scalars ..
      LOGICAL            LQUERY
      CHARACTER          COLBTOP, ROWBTOP
      INTEGER            I, IACOL, IAROW, ICOFF, ICTXT, IINFO, IPW, J,
     $                   JB, JN, LWMIN, MP0, MYCOL, MYROW, NPCOL,
     $                   NPROW, NQ0
*     ..
*     .. Local Arrays ..
      INTEGER            IDUM1( 1 ), IDUM2( 1 )
*     ..
*     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, CHK1MAT, PCHK1MAT, PDGEQR2,
     $                   PDLARFB, PDLARFT, PB_TOPGET, PB_TOPSET, PXERBLA
*     ..
*     .. External Functions ..
      INTEGER            ICEIL, INDXG2P, NUMROC
      EXTERNAL           ICEIL, INDXG2P, NUMROC
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DBLE, MIN, MOD
*     ..
*     .. Executable Statements ..
*
*     Get grid parameters
*
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
*
*     Test the input parameters
*
      INFO = 0
      IF( NPROW.EQ.-1 ) THEN
         INFO = -(600+CTXT_)
      ELSE
         CALL CHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, INFO )
         IF( INFO.EQ.0 ) THEN
            ICOFF = MOD( JA-1, DESCA( NB_ ) )
            IAROW = INDXG2P( IA, DESCA( MB_ ), MYROW, DESCA( RSRC_ ),
     $                       NPROW )
            IACOL = INDXG2P( JA, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ),
     $                       NPCOL )
            MP0 = NUMROC( M+MOD( IA-1, DESCA( MB_ ) ), DESCA( MB_ ),
     $                    MYROW, IAROW, NPROW )
            NQ0 = NUMROC( N+ICOFF, DESCA( NB_ ), MYCOL, IACOL, NPCOL )
            LWMIN = DESCA( NB_ ) * ( MP0 + NQ0 + DESCA( NB_ ) )
*
            WORK( 1 ) = DBLE( LWMIN )
            LQUERY = ( LWORK.EQ.-1 )
            IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY )
     $         INFO = -9
         END IF
         IF( LWORK.EQ.-1 ) THEN
            IDUM1( 1 ) = -1
         ELSE
            IDUM1( 1 ) = 1
         END IF
         IDUM2( 1 ) = 9
         CALL PCHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, 1, IDUM1, IDUM2,
     $                  INFO )
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL PXERBLA( ICTXT, 'PDGEQRF', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
*
*     K = MIN( M, N )
      IPW = DESCA( NB_ ) * DESCA( NB_ ) + 1
      CALL PB_TOPGET( ICTXT, 'Broadcast', 'Rowwise', ROWBTOP )
      CALL PB_TOPGET( ICTXT, 'Broadcast', 'Columnwise', COLBTOP )
      CALL PB_TOPSET( ICTXT, 'Broadcast', 'Rowwise', 'I-ring' )
      CALL PB_TOPSET( ICTXT, 'Broadcast', 'Columnwise', ' ' )
*
*     Handle the first block of columns separately
*
      JN = MIN( ICEIL( JA, DESCA( NB_ ) ) * DESCA( NB_ ), JA+K-1 )
      JB = JN - JA + 1
*
*     Compute the QR factorization of the first block A(ia:ia+m-1,ja:jn)
*
      CALL PDGEQR2( M, JB, A, IA, JA, DESCA, TAU, WORK, LWORK, IINFO )
*
      IF( JA+JB.LE.JA+N-1 ) THEN
*
*        Form the triangular factor of the block reflector
*        H = H(ja) H(ja+1) . . . H(jn)
*
         CALL PDLARFT( 'Forward', 'Columnwise', M, JB, A, IA, JA, DESCA,
     $                 TAU, WORK, WORK( IPW ) )
*
*        Apply H' to A(ia:ia+m-1,ja+jb:ja+n-1) from the left
*
         CALL PDLARFB( 'Left', 'Transpose', 'Forward', 'Columnwise', M,
     $                 N-JB, JB, A, IA, JA, DESCA, WORK, A, IA, JA+JB,
     $                 DESCA, WORK( IPW ) )
      END IF
*
*     Loop over the remaining blocks of columns
*
      DO 10 J = JN+1, JA+K-1, DESCA( NB_ )
         JB = MIN( K-J+JA, DESCA( NB_ ) )
         I = IA + J - JA
*
*        Compute the QR factorization of the current block
*        A(i:ia+m-1,j:j+jb-1)
*
         CALL PDGEQR2( M-J+JA, JB, A, I, J, DESCA, TAU, WORK, LWORK,
     $                 IINFO )
*
         IF( J+JB.LE.JA+N-1 ) THEN
*
*           Form the triangular factor of the block reflector
*           H = H(j) H(j+1) . . . H(j+jb-1)
*
            CALL PDLARFT( 'Forward', 'Columnwise', M-J+JA, JB, A, I, J,
     $                    DESCA, TAU, WORK, WORK( IPW ) )
*
*           Apply H' to A(i:ia+m-1,j+jb:ja+n-1) from the left
*
            CALL PDLARFB( 'Left', 'Transpose', 'Forward', 'Columnwise',
     $                     M-J+JA, N-J-JB+JA, JB, A, I, J, DESCA, WORK,
     $                     A, I, J+JB, DESCA, WORK( IPW ) )
         END IF
*
   10 CONTINUE
*
      CALL PB_TOPSET( ICTXT, 'Broadcast', 'Rowwise', ROWBTOP )
      CALL PB_TOPSET( ICTXT, 'Broadcast', 'Columnwise', COLBTOP )
*
      WORK( 1 ) = DBLE( LWMIN )
*
      RETURN
*
*     End of PDGEQRF
*
      END
