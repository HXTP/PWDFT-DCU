      SUBROUTINE PDGEQP3S( M, N, OFFSET, A, DESCA, G, DESCG,
     $                     VN1, VN2, DESCN, IPIV, TAU, NBIN, NBOUT,
     $                     DWORK, IWORK )
C
C
C     RELEASE $Revision: 1.31 $. COPYRIGHT 2001 (See CONTRIBUTOR).
C
C     PURPOSE
C
C     To compute a step of QR factorization with column pivoting
C     of a real M-by-N matrix A by using Blas-3.
C     It tries to factorize
C     NBIN columns from A starting from the row and column OFFSET+1, and
C     updates all of the matrix with Blas-3 xGEMM.
C
C     In some cases, due to catastrophic cancellations, it cannot
C     factorize NBIN columns.  Hence, the actual number of factorized
C     columns is returned in NBOUT.
C
C     Block A(OFFSET+1:M,OFFSET+1:N) is factorized.
C     Block A(1:OFFSET,OFFSET+1:N) is accordingly pivoted, but not
C     factorized.
C
C     Arguments
C     =========
C
C     M       (input) INTEGER
C             The number of rows of the distributed matrix A. M >= 0.
C
C     N       (input) INTEGER
C             The number of columns of the distributed matrix A. N >= 0
C
C     OFFSET  (input) INTEGER
C             The number of rows and columns of A that have been
C             factorized in previous steps.
C
C     A       (local input/local output) DOUBLE PRECISION pointer into
C             the local memory to an array, dimension (LLD_A,LOCc(N))
C             On entry, the local pieces of the M-by-N distributed
C             matrix A which is to be factored.
C             On exit, a column of NBOUT columns have been factorized,
C             starting in position OFFSET+1. The rest of the matrix is
C             updated.
C
C     DESCA   (global and local input) INTEGER array, dimension DLEN_
C             The array descriptor for the distributed matrix A.
C
C     G       (input/output) DOUBLE PRECISION array, dimension (LDG,N)
C             On entry, a NB-by-N matrix G.
C             On exit, it is overwritten with matrix G.
C
C     DESCG   (global and local input) INTEGER array, dimension DLEN_
C             The array descriptor for the distributed matrix G.
C
C     VN1     (input/output) DOUBLE PRECISION array, dimension (N)
C             The vector with the partial column norms.
C
C     VN2     (input/output) DOUBLE PRECISION array, dimension (N)
C             The vector with the exact column norms.
C
C     DESCN   (global and local input) INTEGER array, dimension DLEN_
C             The array descriptor for the distributed vectors VN1 and
C             VN2.
C
C     IPIV    (input/output) INTEGER array, dimension (N)
C             JPVT(I) = K <=> Column K of the full matrix A has been
C             permuted into position I in AP.
C
C     TAU     (local input/local output) DOUBLE PRECISION, array,
C             dimension LOCc(MIN(M,N)).
C             This array must contain the scalar factors TAU of the
C             elementary reflectors. TAU is tied to the distributed
C             matrix A.
C
C     NBIN    (input) INTEGER
C             The number of columns to factorize.
C
C     NBOUT   (output) INTEGER
C             The number of columns actually factorized.
C
C     Workspace
C
C     DWORK   (local workspace) DOUBLE PRECISION array,
C             dimension ( max( NB, LOCp(M), LOCq(N) )
C
C     IWORK   (local workspace) INTEGER array,
C             dimension (LOCq(N))
C
C     METHOD
C
C     None.
C
C     REFERENCES
C
C     Based on $Revision: 1.31 $ of Parallel Library for Control (PLIC),
C     $Date: 2010/03/24 13:13:28 $, developed by:
C     P. Benner (benner@mathematik.tu-chemnitz.de),
C     E. S. Quintana-Orti (quintana@icc.uji.es), and
C     G. Quintana-Orti (gquintan@icc.uji.es).
C
C     NUMERICAL ASPECTS
C
C     None.
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
C     o Matrices A and Q must be distributed in the same way: same block
C       sizes, same RSCR_, same CSRC_, etc.
C     o The distribution blocks must be square (MB=NB).
C
C     REVISIONS
C
C     None.
C
C     KEYWORDS
C
C     None.
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
      INTEGER            M, N, NBIN, NBOUT, OFFSET
C     ..
C     .. Array Arguments ..
      INTEGER            DESCA( * ), DESCG( * ), DESCN( * ), IPIV( * ),
     $                   IWORK( * )
      DOUBLE PRECISION   A( * ), DWORK( * ), G( * ), TAU( * ), VN1( * ),
     $                   VN2( * )
C     ..
C     .. Local Scalars ..
      INTEGER            IACOL, ICTXT, IOFFA, IPCC, IPCOL, IPR, ITEMP,
     $                   J, JJL, JJPVT, K, KCURCOL, KCURROW, KII, KJJ,
     $                   KP1JJ, LDA, LDG, MP, MYCOL, MYROW, NB, NCC, NJ,
     $                   NJJ, NOC, NPCOL, NPG, NPROW, NQ, PVT
      DOUBLE PRECISION   AKK, ALPHA, BETA, TEMP, TEMP2
C     ..
C     .. Local Arrays ..
      DOUBLE PRECISION   VV( 3 )
C     ..
C     .. External Functions ..
      INTEGER            ICEIL, INDXG2P, NUMROC
      EXTERNAL           ICEIL, INDXG2P, NUMROC
C     ..
C     .. External Subroutines ..
      EXTERNAL           BLACS_GRIDINFO, DCOPY, DGEBR2D, DGEBS2D,
     $                   DGERV2D, DGESD2D, DLARFG, DSCAL, DSWAP,
     $                   IGERV2D, IGESD2D, IGSUM2D, INFOG1L, INFOG2L,
     $                   PDAMAX, PDELSET, PDGEMM, PDGEMV, PDLARFG,
     $                   PDLASET, PDNRM2
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC          ABS, DBLE, IDINT, MAX, MIN, SQRT
C     ..
C     .. Executable Statements ..
C
C     Quick return if possible.
C
      NBOUT = 0
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
C
C     Get grid parameters and initialize some variables.
C
      ICTXT = DESCA( CTXT_ )
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      MP = NUMROC( M, DESCA( MB_ ), MYROW, DESCA( RSRC_ ), NPROW )
      NQ = NUMROC( N, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
      LDA = DESCA( LLD_ )
      LDG = DESCG( LLD_ )
      NB = NBIN
      NPG = NUMROC( NB, DESCG( MB_ ), MYROW, DESCG( RSRC_ ), NPROW )
C
C     Initializations.
C
      IPR  = 1
C
      IPCC = 1
C
C     Set G to zero.
C
      CALL PDLASET( 'All', NB, N, ZERO, ZERO, G, 1, 1, DESCG )
C
C     *************************
C     * Compute factorization *
C     *************************
C
      J = 1
      NCC = 0
C
 1000 IF ( ( J.LE.NBIN ).AND.( NCC.EQ.0 ) ) THEN
C
        K = OFFSET+J
        CALL INFOG2L( K, K, DESCA, NPROW, NPCOL, MYROW, MYCOL,
     $                KII, KJJ, KCURROW, KCURCOL )
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as101', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as201', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Determining column with largest two-norm.
C       =========================================
C
        NOC = N-K+1
        IF( NOC.GT.1 ) THEN
          CALL PDAMAX( NOC, TEMP, PVT, VN1, 1, K, DESCN, DESCN( M_ ) )
        ELSE
          PVT = K
        END IF
C       PVT = K
C
C       Pivoting: matrix A, matrix G, norm vector and ipiv vector.
C       ==========================================================
C
        IF( K.NE.PVT ) THEN
          CALL INFOG1L( PVT, DESCA( NB_ ), NPCOL, MYCOL,
     $                  DESCA( CSRC_ ), JJPVT, IPCOL )
          IF( KCURCOL.EQ.IPCOL ) THEN
            IF( MYCOL.EQ.KCURCOL ) THEN
              CALL DSWAP( MP, A( 1+(KJJ-1)*LDA ), 1,
     $                        A( 1+(JJPVT-1)*LDA ), 1 )
              CALL DSWAP( NB, G( 1+(KJJ-1)*LDG ), 1,
     $                        G( 1+(JJPVT-1)*LDG ), 1 )
              ITEMP = IPIV( JJPVT )
              IPIV( JJPVT ) = IPIV( KJJ )
              IPIV( KJJ ) = ITEMP
              VN1( JJPVT ) = VN1( KJJ )
              VN2( JJPVT ) = VN2( KJJ )
            END IF
          ELSE
            IF( MYCOL.EQ.KCURCOL ) THEN
C
CCCC          print *, ' Intercambiando datos...', mp
              CALL DGESD2D( ICTXT, MP, 1, A( 1+(KJJ-1)*LDA ), LDA,
     $                      MYROW, IPCOL )
              CALL DGERV2D( ICTXT, MP, 1, A( 1+(KJJ-1)*LDA ), LDA,
     $                      MYROW, IPCOL )
C
              IF( NPG.GT.0 ) THEN
                CALL DGESD2D( ICTXT, NPG, 1, G( 1+(KJJ-1)*LDG ), LDG,
     $                        MYROW, IPCOL )
                CALL DGERV2D( ICTXT, NPG, 1, G( 1+(KJJ-1)*LDG ), LDG,
     $                        MYROW, IPCOL )
              END IF
C
              VV( 1 ) = DBLE( IPIV( KJJ ) )
              VV( 2 ) = VN1( KJJ )
              VV( 3 ) = VN2( KJJ )
              CALL DGESD2D( ICTXT, 3, 1, VV( 1 ), 3, MYROW,
     $                      IPCOL )
              CALL IGERV2D( ICTXT, 1, 1, IPIV( KJJ ), 1, MYROW,
     $                      IPCOL )
C
            ELSE IF( MYCOL.EQ.IPCOL ) THEN
C
              CALL DGERV2D( ICTXT, MP, 1, DWORK( IPR ), 1,
     $                      MYROW, KCURCOL )
              CALL DGESD2D( ICTXT, MP, 1, A( 1+(JJPVT-1)*LDA ), LDA,
     $                      MYROW, KCURCOL )
              CALL DCOPY( MP, DWORK( IPR ), 1, A( 1+(JJPVT-1)*LDA ), 1 )
C
              IF( NPG.GT.0 ) THEN
                CALL DGERV2D( ICTXT, NPG, 1, DWORK( IPR ), 1,
     $                        MYROW, KCURCOL )
                CALL DGESD2D( ICTXT, NPG, 1, G( 1+(JJPVT-1)*LDG ),
     $                        LDG, MYROW, KCURCOL )
                CALL DCOPY( NPG, DWORK( IPR ), 1,
     $                           G( 1+(JJPVT-1)*LDG ), 1 )
              END IF
C
              CALL DGERV2D( ICTXT, 3, 1, VV( 1 ), 3, MYROW,
     $                      KCURCOL )
              CALL IGESD2D( ICTXT, 1, 1, IPIV( JJPVT ), 1, MYROW,
     $                      KCURCOL )
              IPIV( JJPVT ) = IDINT( VV( 1 ) )
              VN1( JJPVT ) = VV( 2 )
              VN2( JJPVT ) = VV( 3 )
C
            END IF
C
          END IF
C
        END IF
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as102', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as202', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Update pivot column.
C       ====================
C
        IF( J.GT.1 ) THEN
C
C         af( k:m, k ) = af( k:m, k ) - ...
C                        af( k:m, offset+1:k-1 ) * g( 1:j-1, k );
C
          CALL PDGEMV( 'No transpose', M-K+1, J-1,
     $                 -ONE, A, K, OFFSET+1, DESCA,
     $                       G, 1, K, DESCG, 1,
     $                 ONE, A, K, K, DESCA, 1 )
        END IF
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as103', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as203', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Compute Householder transformation.
C       ===================================
C
C       [ v, betaf(k), diag ] = housepla( af( k:m, k ) );
C       betaf(k) = - betaf(k);
C
        IF( DESCA( M_ ).EQ.1 ) THEN
          IF( MYROW.EQ.KCURROW ) THEN
            IF( MYCOL.EQ.KCURCOL ) THEN
              IOFFA = KII+(KJJ-1)*LDA
              AKK = A( IOFFA )
              CALL DLARFG( 1, AKK, A( IOFFA ), 1, TAU( KJJ ) )
              IF( N.GT.1 ) THEN
                ALPHA = ONE - TAU( KJJ )
                CALL DGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                        1 )
                CALL DSCAL( NQ-KJJ, ALPHA, A( IOFFA+LDA ), LDA )
              END IF
              CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, 1,
     $                      TAU( KJJ ), 1 )
              A( IOFFA ) = AKK
            ELSE
              IF( N.GT.1 ) THEN
                CALL DGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA,
     $                        1, KCURROW, KCURCOL )
                CALL DSCAL( NQ-KJJ+1, ALPHA, A( K ), LDA )
              END IF
            END IF
          ELSE IF( MYCOL.EQ.KCURCOL ) THEN
            CALL DGEBR2D( ICTXT, 'Columnwise', ' ', 1, 1, TAU( KJJ ),
     $                    1, KCURROW, KCURCOL )
          END IF
C
        ELSE
C
          CALL PDLARFG( M-K+1, AKK, K, K, A, MIN( K+1, M ), K,
     $                  DESCA, 1, TAU )
C
        END IF
C
        CALL PDELSET( A, K, K, DESCA, ONE )
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as104', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as204', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Incremental computation of G.
C       =============================
C
        IF( J.LT.N ) THEN
C
C         g( j, k+1:n ) = betaf(k) * af( k:m, k )' * af( k:m, k+1:n );
C
C         Broadcasting beta.
C
          IF( MYCOL.EQ.KCURCOL ) THEN
            BETA = TAU( KJJ )
            CALL DGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, BETA, 1 )
          ELSE
            CALL DGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, BETA, 1,
     $                    MYROW, KCURCOL )
          END IF
C
          CALL PDGEMV( 'Transpose', M-K+1, N-K,
     $                 BETA, A, K, K+1, DESCA,
     $                       A, K, K, DESCA, 1,
     $                 ZERO, G, J, K+1, DESCG, DESCG( M_ ) )
        END IF
C
        IF( J.GT.1 ) THEN
C
C         g( j, offset+1:n ) = g( j, offset+1:n ) -  betaf(k) * ...
C                    af( k:m, k )' * af( k:m, offset+1:k-1 ) * ...
C                    g( 1:j-1, offset+1:n );
C
          CALL PDGEMV( 'Transpose', M-K+1, J-1,
     $                 -BETA, A, K, OFFSET+1, DESCA,
     $                        A, K, K, DESCA, 1,
     $                 ZERO, G, 1, 1, DESCG, 1 )
          CALL PDGEMV( 'Transpose', J-1, N-OFFSET,
     $                 ONE, G, 1, OFFSET+1, DESCG,
     $                      G, 1, 1, DESCG, 1,
     $                 ONE, G, J, OFFSET+1, DESCG, DESCG( M_ ) )
        END IF
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as105', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs105', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as205', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs205', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Update pivot row.
C       =================
C
C       af( k, k+1:n ) = af( k, k+1:n ) - ...
C                        af( k, offset+1:k ) * g( 1:j, k+1:n );
C
        CALL PDGEMV( 'Transpose', J, N-K,
     $               -ONE, G, 1, K+1, DESCG,
     $                     A, K, OFFSET+1, DESCA, DESCA( M_ ),
     $               ONE, A, K, K+1, DESCA, DESCA( M_ ) )
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as106', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs106', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as206', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs206', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Restore diagonal value.
C       =======================
C
C       af( k, k ) = akk;
C
        CALL PDELSET( A, K, K, DESCA, AKK )
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as107', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs107', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as207', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs207', 6, DWORK( 50000 ) )
CCCC    END IF
C
C       Norm downdate.
C       ==============
C
C       Broadcast columnwise elements (K,K+1:N) of A.
C
CCCC    print *, myrow, mycol, ' before downdating: ', j, ' ',
CCCC $                          vn1(1),vn1(2),vn1(3),vn1(4)
        IF( MYCOL.EQ.KCURCOL ) THEN
          KP1JJ = KJJ + 1
        ELSE
          KP1JJ = KJJ
        END IF
C
        IF( MYROW.EQ.KCURROW ) THEN
          CALL DCOPY( NQ-KP1JJ+1, A( KII+( MIN( NQ, KP1JJ )-1)*LDA ),
     $                LDA, DWORK( IPR+MIN( NQ, KP1JJ )-1 ), 1 )
          CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, NQ-KP1JJ+1,
     $                  DWORK( IPR+MIN( NQ, KP1JJ )-1 ), 1 )
        ELSE
          CALL DGEBR2D( ICTXT, 'Columnwise', ' ', NQ-KP1JJ+1, 1,
     $                  DWORK( IPR+MIN( NQ, KP1JJ )-1 ),
     $                  MAX( 1, NQ ), KCURROW, MYCOL )
        END IF
C
C       Initialize vector with cancellations.
C
        DO JJL = KP1JJ, NQ
          IWORK( IPCC-1+JJL ) = 0
        END DO
C
C       Loop for updating the norms in K+1..N.
C
        DO NJ = K+1, N
C
          CALL INFOG1L( NJ, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ),
     $                  NJJ, IACOL )
C
          IF( MYCOL.EQ.IACOL ) THEN
C
             IF( VN1( NJJ ).NE.ZERO ) THEN
               TEMP = ONE-( ABS( DWORK( IPR-1+NJJ ) ) /
     $                      VN1( NJJ ) )**2
               TEMP = MAX( TEMP, ZERO )
               TEMP2 = ONE + 0.05D+0*TEMP*
     $               ( VN1( NJJ ) / VN2( NJJ ) )**2
               IF( TEMP2.EQ.ONE ) THEN
C
C                Norm should be recomputed: Blocking finishes.
C
                 NCC = NCC+1
                 IWORK( IPCC-1+NJJ ) = NJ
C
               ELSE
C
C                Do not recompute norm.
C
                 VN1( NJJ ) = VN1( NJJ ) * SQRT( TEMP )
               END IF
             END IF
C
          ELSE
C
          END IF
C
        END DO
CCCC    print *, myrow, mycol, ' after downdating: ', j, ' ',
CCCC $                          vn1(1),vn1(2),vn1(3),vn1(4)
C
C       Adding variables NCC.
C
        CALL IGSUM2D( ICTXT, 'All', ' ', 1, 1, NCC, 1, -1, 0 )
C
CCCC    if( myrow.eq.0 .and. mycol.eq.0 ) then
CCCC      IF( J.EQ.1 ) THEN
CCCC        print *, ' it1NCC = ', ncc, ' ; '
CCCC      ELSEIF( J.EQ.2 ) THEN
CCCC        print *, ' it2NCC = ', ncc, ' ; '
CCCC      END IF
CCCC    end if
CCCC    IF( J.EQ.1 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as108', 6, DWORK( 50000 ) )
CCCC    ELSEIF( J.EQ.2 ) THEN
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as208', 6, DWORK( 50000 ) )
CCCC    END IF
C
        J = J+1
        GO TO 1000
C
      END IF
C
      NBOUT = J-1
      NB    = NBOUT
C
C     Block update.
C     =============
C
      IF( K.LT.M ) THEN
C
C       af( k+1:m, offset+nb+1:n ) = af( k+1:m, offset+nb+1:n ) - ...
C           af( k+1:m, offset+1:offset+nb ) * g( 1:nb, offset+nb+1:n );
C
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as10', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs10', 6, DWORK( 50000 ) )
        CALL PDGEMM( 'No transpose', 'No transpose',
     $               M-K, N-OFFSET-NB, NB,
     $               -ONE, A, K+1, OFFSET+1, DESCA,
     $                     G, 1, OFFSET+NB+1, DESCG,
     $               ONE, A, K+1, OFFSET+NB+1, DESCA )
      END IF
CCCC      CALL PDLAPRN2( M, N, A, 1, 1, DESCA, 0, 0,
CCCC $                   'as11', 6, DWORK( 50000 ) )
CCCC      CALL PDLAPRN2( NB, N, G, 1, 1, DESCG, 0, 0,
CCCC $                   'gs11', 6, DWORK( 50000 ) )
C
C     Recomputation of norms.
C     =======================
C
      IF( NCC.GT.0 ) THEN
C
        DO JJL = 1, NQ
C
          NJ = IWORK( IPCC-1+JJL )
C
          IF( NJ.NE.0 ) THEN
            IF( M.GT.K ) THEN
              CALL PDNRM2( M-K, VN1( JJL ), A, K+1, NJ, DESCA, 1 )
              VN2( JJL ) = VN1( JJL )
            ELSE
              VN1( JJL ) = ZERO
              VN2( JJL ) = ZERO
            END IF
          END IF

        END DO
C
      END IF
C
      RETURN
C
C *** Last line of PDGEQP3S ***
      END
