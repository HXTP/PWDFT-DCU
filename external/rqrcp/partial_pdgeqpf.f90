! A PARTIAL VERSION OF PDGEQPF
! AUTHOR: JIANWEI XIAO

SUBROUTINE PARTIAL_PDGEQPF( PARTIAL, M, N, A, IA, JA, DESCA, IPIV, TAU, WORK, LWORK, INFO )
!     .. Scalar Arguments ..
     INTEGER            IA, JA, INFO, LWORK, M, N, PARTIAL
!     ..
!     .. Array Arguments ..
     INTEGER            DESCA( * ), IPIV( * )
     DOUBLE PRECISION   A( * ), TAU( * ), WORK( * )
!     .. Parameters ..
     INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DTYPE_, LLD_, MB_, M_, NB_, N_, RSRC_
     PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DTYPE_ = 1, CTXT_ = 2, M_ = 3, &
          N_ = 4, MB_ = 5, NB_ = 6, RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
     DOUBLE PRECISION   ONE, ZERO
     PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
!     ..
!     .. Local Scalars ..
     LOGICAL            LQUERY
     INTEGER            I, IACOL, IAROW, ICOFF, ICTXT, ICURROW, ICURCOL, II, IIA, IOFFA, &
     IPN, IPCOL, IPW, IROFF, ITEMP, J, JB, JJ, JJA, JJPVT, JN, KB, K, KK, KSTART, KSTEP, &
     LDA, LL, LWMIN, MN, MP, MYCOL, MYROW, NPCOL, NPROW, NQ, NQ0, PVT
     DOUBLE PRECISION   AJJ, ALPHA, TEMP, TEMP2
!     ..
!     .. Local Arrays ..
     INTEGER            DESCN( DLEN_ ), IDUM1( 1 ), IDUM2( 1 )
!     ..
!     .. External Subroutines ..
     EXTERNAL           BLACS_GRIDINFO, CHK1MAT, DCOPY, DESCSET, DGEBR2D, DGEBS2D, DGERV2D, &
     DGESD2D, DLARFG, DSWAP, IGERV2D, IGESD2D, INFOG1L, INFOG2L, PCHK1MAT, PDAMAX, PDELSET, &
     PDLARF, PDLARFG, PDNRM2, PXERBLA
!     ..
!     .. External Functions ..
     INTEGER            ICEIL, INDXG2P, NUMROC
     EXTERNAL           ICEIL, INDXG2P, NUMROC
!     ..
!     .. Intrinsic Functions ..
     INTRINSIC          ABS, DBLE, IDINT, MAX, MIN, MOD, SQRT
!     ..
!     .. Executable Statements ..
!
!     Get grid parameters
!
     ICTXT = DESCA( CTXT_ )
     CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
!
!     Test the input parameters
!
     INFO = 0
     IF( NPROW.EQ.-1 ) THEN
          INFO = -(600+CTXT_)
     ELSE
          CALL CHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, INFO )
          IF( INFO.EQ.0 ) THEN
               IROFF = MOD( IA-1, DESCA( MB_ ) )
               ICOFF = MOD( JA-1, DESCA( NB_ ) )
               IAROW = INDXG2P( IA, DESCA( MB_ ), MYROW, DESCA( RSRC_ ), NPROW )
               IACOL = INDXG2P( JA, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
               MP = NUMROC( M+IROFF, DESCA( MB_ ), MYROW, IAROW, NPROW )
               NQ = NUMROC( N+ICOFF, DESCA( NB_ ), MYCOL, IACOL, NPCOL )
               NQ0 = NUMROC( JA+N-1, DESCA( NB_ ), MYCOL, DESCA( CSRC_ ), NPCOL )
               LWMIN = MAX( 3, MP + NQ ) + NQ0 + NQ
               WORK( 1 ) = DBLE( LWMIN )
               LQUERY = ( LWORK.EQ.-1 )
               IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY ) THEN
                    INFO = -10
               ELSE
               END IF
          ELSE
          END IF
          IF( LWORK.EQ.-1 ) THEN
               IDUM1( 1 ) = -1
          ELSE
               IDUM1( 1 ) = 1
          END IF
          IDUM2( 1 ) = 10
          CALL PCHK1MAT( M, 1, N, 2, IA, JA, DESCA, 6, 1, IDUM1, IDUM2, INFO )
     END IF

     IF( INFO.NE.0 ) THEN
          CALL PXERBLA( ICTXT, 'PDGEQPF', -INFO )
          RETURN
     ELSE IF( LQUERY ) THEN
          RETURN
     END IF
!
!     Quick return if possible
!
     IF( M.EQ.0 .OR. N.EQ.0 ) THEN
          RETURN
     ELSE
     END IF
!
     CALL INFOG2L( IA, JA, DESCA, NPROW, NPCOL, MYROW, MYCOL, IIA, JJA, IAROW, IACOL )
     IF( MYROW.EQ.IAROW ) THEN
          MP = MP - IROFF
     ELSE
     END IF
     IF( MYCOL.EQ.IACOL ) THEN
          NQ = NQ - ICOFF
     ELSE
     END IF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !MN = MIN( M, N )
      MN = PARTIAL
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     Initialize the array of pivots
!
     LDA = DESCA( LLD_ )
     JN = MIN( ICEIL( JA, DESCA( NB_ ) ) * DESCA( NB_ ), JA+N-1 )
     KSTEP  = NPCOL * DESCA( NB_ )
!
     IF( MYCOL.EQ.IACOL ) THEN
!
!        Handle first block separately
!
          JB = JN - JA + 1
          DO LL = JJA, JJA+JB-1
               IPIV( LL ) = JA + LL - JJA
          END DO
          KSTART = JN + KSTEP - DESCA( NB_ )
!
!        Loop over remaining block of columns
!
          DO KK = JJA+JB, JJA+NQ-1, DESCA( NB_ )
               KB = MIN( JJA+NQ-KK, DESCA( NB_ ) )
               DO LL = KK, KK+KB-1
                    IPIV( LL ) = KSTART+LL-KK+1
               END DO
               KSTART = KSTART + KSTEP
          END DO
     ELSE
          KSTART = JN + ( MOD( MYCOL-IACOL+NPCOL, NPCOL )-1 )*DESCA( NB_ )
          DO KK = JJA, JJA+NQ-1, DESCA( NB_ )
               KB = MIN( JJA+NQ-KK, DESCA( NB_ ) )
               DO LL = KK, KK+KB-1
                    IPIV( LL ) = KSTART+LL-KK+1
               END DO
               KSTART = KSTART + KSTEP
          END DO
     END IF
!
!     Initialize partial column norms, handle first block separately
!
     CALL DESCSET( DESCN, 1, DESCA( N_ ), 1, DESCA( NB_ ), MYROW, DESCA( CSRC_ ), ICTXT, 1 )
!
     IPN = 1
     IPW = IPN + NQ0 + NQ
     JJ = IPN + JJA - 1
     IF( MYCOL.EQ.IACOL ) THEN
          DO KK = 0, JB-1
               CALL PDNRM2( M, WORK( JJ+KK ), A, IA, JA+KK, DESCA, 1 )
               WORK( NQ+JJ+KK ) = WORK( JJ+KK )
          END DO
          JJ = JJ + JB
     END IF
     ICURCOL = MOD( IACOL+1, NPCOL )
!
!     Loop over the remaining blocks of columns
!
     DO J = JN+1, JA+N-1, DESCA( NB_ )
          JB = MIN( JA+N-J, DESCA( NB_ ) )
!
          IF( MYCOL.EQ.ICURCOL ) THEN
               DO KK = 0, JB-1
                    CALL PDNRM2( M, WORK( JJ+KK ), A, IA, J+KK, DESCA, 1 )
                    WORK( NQ+JJ+KK ) = WORK( JJ+KK )
               END DO
               JJ = JJ + JB
          END IF
          ICURCOL = MOD( ICURCOL+1, NPCOL )
     END DO
!
!     Compute factorization
!
     DO J = JA, JA+MN-1
          I = IA + J - JA
!
          CALL INFOG1L( J, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ), JJ, ICURCOL )
          K = JA + N - J
          IF( K.GT.1 ) THEN
               CALL PDAMAX( K, TEMP, PVT, WORK( IPN ), 1, J, DESCN, DESCN( M_ ) )
          ELSE
               PVT = J
          END IF
          IF( J.NE.PVT ) THEN
               CALL INFOG1L( PVT, DESCA( NB_ ), NPCOL, MYCOL, DESCA( CSRC_ ), JJPVT, IPCOL )
               IF( ICURCOL.EQ.IPCOL ) THEN
                    IF( MYCOL.EQ.ICURCOL ) THEN
                         CALL DSWAP( MP, A( IIA+(JJ-1)*LDA ), 1, A( IIA+(JJPVT-1)*LDA ), 1 )
                         ITEMP = IPIV( JJPVT )
                         IPIV( JJPVT ) = IPIV( JJ )
                         IPIV( JJ ) = ITEMP
                         WORK( IPN+JJPVT-1 ) = WORK( IPN+JJ-1 )
                         WORK( IPN+NQ+JJPVT-1 ) = WORK( IPN+NQ+JJ-1 )
                    END IF
               ELSE
                    IF( MYCOL.EQ.ICURCOL ) THEN
!
                         CALL DGESD2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA, MYROW, IPCOL )
                         WORK( IPW )   = DBLE( IPIV( JJ ) )
                         WORK( IPW+1 ) = WORK( IPN + JJ - 1 )
                         WORK( IPW+2 ) = WORK( IPN + NQ + JJ - 1 )
                         CALL DGESD2D( ICTXT, 3, 1, WORK( IPW ), 3, MYROW, IPCOL )
!
                         CALL DGERV2D( ICTXT, MP, 1, A( IIA+(JJ-1)*LDA ), LDA, MYROW, IPCOL )
                         CALL IGERV2D( ICTXT, 1, 1, IPIV( JJ ), 1, MYROW, IPCOL )
!
                    ELSE IF( MYCOL.EQ.IPCOL ) THEN
!
                         CALL DGESD2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ), LDA, MYROW, ICURCOL )
                         CALL IGESD2D( ICTXT, 1, 1, IPIV( JJPVT ), 1, MYROW, ICURCOL )
!
                         CALL DGERV2D( ICTXT, MP, 1, A( IIA+(JJPVT-1)*LDA ), LDA, MYROW, ICURCOL )
                         CALL DGERV2D( ICTXT, 3, 1, WORK( IPW ), 3, MYROW, ICURCOL )
                         IPIV( JJPVT ) = IDINT( WORK( IPW ) )
                         WORK( IPN+JJPVT-1 ) = WORK( IPW+1 )
                         WORK( IPN+NQ+JJPVT-1 ) = WORK( IPW+2 )
!
                    END IF
!
               END IF
!
          END IF
!
!        Generate elementary reflector H(i)
!
          CALL INFOG1L( I, DESCA( MB_ ), NPROW, MYROW, DESCA( RSRC_ ), II, ICURROW )
          IF( DESCA( M_ ).EQ.1 ) THEN
               IF( MYROW.EQ.ICURROW ) THEN
                    IF( MYCOL.EQ.ICURCOL ) THEN
                         IOFFA = II+(JJ-1)*DESCA( LLD_ )
                         AJJ = A( IOFFA )
                         CALL DLARFG( 1, AJJ, A( IOFFA ), 1, TAU( JJ ) )
                         IF( N.GT.1 ) THEN
                              ALPHA = ONE - TAU( JJ )
                              CALL DGEBS2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA, 1 )
                              CALL DSCAL( NQ-JJ, ALPHA, A( IOFFA+DESCA( LLD_ ) ), DESCA( LLD_ ) )
                         END IF
                         CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, 1, TAU( JJ ), 1 )
                         A( IOFFA ) = AJJ
                    ELSE
                         IF( N.GT.1 ) THEN
                              CALL DGEBR2D( ICTXT, 'Rowwise', ' ', 1, 1, ALPHA, 1, ICURROW, ICURCOL )
                              CALL DSCAL( NQ-JJ+1, ALPHA, A( I ), DESCA( LLD_ ) )
                         END IF
                    END IF
               ELSE IF( MYCOL.EQ.ICURCOL ) THEN
                    CALL DGEBR2D( ICTXT, 'Columnwise', ' ', 1, 1, TAU( JJ ), 1, ICURROW, ICURCOL )
               END IF
!
          ELSE
!
               CALL PDLARFG( M-J+JA, AJJ, I, J, A, MIN( I+1, IA+M-1 ), J, DESCA, 1, TAU )
               IF( J.LT.JA+N-1 ) THEN
!
!              Apply H(i) to A(ia+j-ja:ia+m-1,j+1:ja+n-1) from the left
!
                    CALL PDELSET( A, I, J, DESCA, ONE )
                    CALL PDLARF( 'Left', M-J+JA, JA+N-1-J, A, I, J, DESCA, 1, TAU, A, I, J+1, DESCA, WORK( IPW ) )
               END IF
               CALL PDELSET( A, I, J, DESCA, AJJ )
!
          END IF
!
!        Update partial columns norms
!
          IF( MYCOL.EQ.ICURCOL ) THEN
               JJ = JJ + 1
          ELSE
          END IF
          IF( MOD( J, DESCA( NB_ ) ).EQ.0 ) THEN
               ICURCOL = MOD( ICURCOL+1, NPCOL )
          ELSE
          END IF
          IF( (JJA+NQ-JJ).GT.0 ) THEN
               IF( MYROW.EQ.ICURROW ) THEN
                    CALL DGEBS2D( ICTXT, 'Columnwise', ' ', 1, JJA+NQ-JJ, &
                         A( II+( MIN( JJA+NQ-1, JJ )-1 )*LDA ), LDA )
                    CALL DCOPY( JJA+NQ-JJ, A( II+( MIN( JJA+NQ-1, JJ )-1)*LDA ), &
                         LDA, WORK( IPW+MIN( JJA+NQ-1,JJ )-1 ), 1 )
               ELSE
                    CALL DGEBR2D( ICTXT, 'Columnwise', ' ', JJA+NQ-JJ, 1, &
                         WORK( IPW+MIN( JJA+NQ-1, JJ )-1 ), MAX( 1, NQ ), ICURROW, MYCOL )
               END IF
          END IF
!
          JN = MIN( ICEIL( J+1, DESCA( NB_ ) ) * DESCA( NB_ ), JA + N - 1 )
          IF( MYCOL.EQ.ICURCOL ) THEN
               DO LL = JJ-1, JJ + JN - J - 2
                    IF( WORK( IPN+LL ).NE.ZERO ) THEN
                         TEMP = ONE-( ABS( WORK( IPW+LL ) ) / WORK( IPN+LL ) )**2
                         TEMP = MAX( TEMP, ZERO )
                         TEMP2 = ONE + 0.05D+0*TEMP*( WORK( IPN+LL ) / WORK( IPN+NQ+LL ) )**2
                         IF( TEMP2.EQ.ONE ) THEN
                              IF( IA+M-1.GT.I ) THEN
                                   CALL PDNRM2( IA+M-I-1, WORK( IPN+LL ), A, I+1, J+LL-JJ+2, DESCA, 1 )
                                   WORK( IPN+NQ+LL ) = WORK( IPN+LL )
                              ELSE
                                   WORK( IPN+LL ) = ZERO
                                   WORK( IPN+NQ+LL ) = ZERO
                              END IF
                         ELSE
                              WORK( IPN+LL ) = WORK( IPN+LL ) * SQRT( TEMP )
                         END IF
                    END IF
               END DO
               JJ = JJ + JN - J
          END IF
          ICURCOL = MOD( ICURCOL+1, NPCOL )
!
          DO K = JN+1, JA+N-1, DESCA( NB_ )
               KB = MIN( JA+N-K, DESCA( NB_ ) )
!
               IF( MYCOL.EQ.ICURCOL ) THEN
                    DO LL = JJ-1, JJ+KB-2
                         IF( WORK( IPN+LL ).NE.ZERO ) THEN
                              TEMP = ONE-( ABS( WORK( IPW+LL ) ) / WORK( IPN+LL ) )**2
                              TEMP = MAX( TEMP, ZERO )
                              TEMP2 = ONE + 0.05D+0*TEMP*( WORK( IPN+LL ) / WORK( IPN+NQ+LL ) )**2
                              IF( TEMP2.EQ.ONE ) THEN
                                   IF( IA+M-1.GT.I ) THEN
                                        CALL PDNRM2( IA+M-I-1, WORK( IPN+LL ), A, I+1, K+LL-JJ+1, DESCA, 1 )
                                        WORK( IPN+NQ+LL ) = WORK( IPN+LL )
                                   ELSE
                                        WORK( IPN+LL ) = ZERO
                                        WORK( IPN+NQ+LL ) = ZERO
                                   END IF
                              ELSE
                                   WORK( IPN+LL ) = WORK( IPN+LL ) * SQRT( TEMP )
                              END IF
                         END IF
                    END DO
                    JJ = JJ + KB
               END IF
               ICURCOL = MOD( ICURCOL+1, NPCOL )
!
          END DO
!
     END DO
!
     WORK( 1 ) = DBLE( LWMIN )
!
     RETURN
!
!     End of PDGEQPF
!
END SUBROUTINE PARTIAL_PDGEQPF
