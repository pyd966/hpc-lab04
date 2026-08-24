! Two-field lopsided advection prototype.
!
! The interior path shares the beta sign coefficients between two independent
! fields.  Boundary points are delegated to lopsided(..., shell_only=.true.)
! so the established boundary and symmetry rules remain the single source of
! truth.  The caller enables this routine only for an explicit A/B experiment.
subroutine lopsided2(ex,X,Y,Z,f1,f1_rhs,f2,f2_rhs,Sfx,Sfy,Sfz,Symmetry,SoA1,SoA2)
  implicit none
  integer, intent(in) :: ex(1:3), Symmetry
  real*8, intent(in) :: X(1:ex(1)), Y(1:ex(2)), Z(1:ex(3))
  real*8, dimension(ex(1),ex(2),ex(3)), intent(in) :: f1, f2, Sfx, Sfy, Sfz
  real*8, dimension(ex(1),ex(2),ex(3)), intent(inout) :: f1_rhs, f2_rhs
  real*8, dimension(3), intent(in) :: SoA1, SoA2
  real*8, dimension(-2:ex(1),-2:ex(2),-2:ex(3)) :: fh1, fh2
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  integer :: ibegin,iend,jbegin,jend,kbegin,kend
  real*8 :: dX,dY,dZ,d12dx,d12dy,d12dz
  real*8 :: bxp,bxn,byp,byn,bzp,bzn
  real*8 :: ax,ay,az
  real*8, parameter :: ZEO=0.d0, F3=3.d0, F6=6.d0, F10=10.d0
  real*8, parameter :: F18=18.d0, F12=12.d0
  integer, parameter :: NO_SYMM=0, EQ_SYMM=1

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)
  d12dx = ZEO + 1.d0/F12/dX
  d12dy = ZEO + 1.d0/F12/dY
  d12dz = ZEO + 1.d0/F12/dZ
  imax=ex(1); jmax=ex(2); kmax=ex(3)
  imin=1; jmin=1; kmin=1
  if (Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin=-2
  if (Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin=-2
  if (Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin=-2

  call symmetry_bd(3,ex,f1,fh1,SoA1)
  call symmetry_bd(3,ex,f2,fh2,SoA2)

#ifdef AMSS_LOPSIDEDIFF_SIMD
  ibegin=max(1,imin+3); iend=min(ex(1)-1,imax-3)
  jbegin=max(1,jmin+3); jend=min(ex(2)-1,jmax-3)
  kbegin=max(1,kmin+3); kend=min(ex(3)-1,kmax-3)
  if (ibegin <= iend .and. jbegin <= jend .and. kbegin <= kend) then
    do k=kbegin,kend
    do j=jbegin,jend
!$omp simd
    do i=ibegin,iend
      ! Load each beta sign coefficient once for both fields.
      bxp = max(Sfx(i,j,k),ZEO)*d12dx
      bxn = min(Sfx(i,j,k),ZEO)*d12dx
      byp = max(Sfy(i,j,k),ZEO)*d12dy
      byn = min(Sfy(i,j,k),ZEO)*d12dy
      bzp = max(Sfz(i,j,k),ZEO)*d12dz
      bzn = min(Sfz(i,j,k),ZEO)*d12dz
      ax = bxp*(-F3*fh1(i-1,j,k)-F10*fh1(i,j,k)+F18*fh1(i+1,j,k)-F6*fh1(i+2,j,k)+fh1(i+3,j,k)) &
          - bxn*(-F3*fh1(i+1,j,k)-F10*fh1(i,j,k)+F18*fh1(i-1,j,k)-F6*fh1(i-2,j,k)+fh1(i-3,j,k))
      ay = byp*(-F3*fh1(i,j-1,k)-F10*fh1(i,j,k)+F18*fh1(i,j+1,k)-F6*fh1(i,j+2,k)+fh1(i,j+3,k)) &
          - byn*(-F3*fh1(i,j+1,k)-F10*fh1(i,j,k)+F18*fh1(i,j-1,k)-F6*fh1(i,j-2,k)+fh1(i,j-3,k))
      az = bzp*(-F3*fh1(i,j,k-1)-F10*fh1(i,j,k)+F18*fh1(i,j,k+1)-F6*fh1(i,j,k+2)+fh1(i,j,k+3)) &
          - bzn*(-F3*fh1(i,j,k+1)-F10*fh1(i,j,k)+F18*fh1(i,j,k-1)-F6*fh1(i,j,k-2)+fh1(i,j,k-3))
      f1_rhs(i,j,k)=f1_rhs(i,j,k)+ax
      f1_rhs(i,j,k)=f1_rhs(i,j,k)+ay
      f1_rhs(i,j,k)=f1_rhs(i,j,k)+az
      ax = bxp*(-F3*fh2(i-1,j,k)-F10*fh2(i,j,k)+F18*fh2(i+1,j,k)-F6*fh2(i+2,j,k)+fh2(i+3,j,k)) &
          - bxn*(-F3*fh2(i+1,j,k)-F10*fh2(i,j,k)+F18*fh2(i-1,j,k)-F6*fh2(i-2,j,k)+fh2(i-3,j,k))
      ay = byp*(-F3*fh2(i,j-1,k)-F10*fh2(i,j,k)+F18*fh2(i,j+1,k)-F6*fh2(i,j+2,k)+fh2(i,j+3,k)) &
          - byn*(-F3*fh2(i,j+1,k)-F10*fh2(i,j,k)+F18*fh2(i,j-1,k)-F6*fh2(i,j-2,k)+fh2(i,j-3,k))
      az = bzp*(-F3*fh2(i,j,k-1)-F10*fh2(i,j,k)+F18*fh2(i,j,k+1)-F6*fh2(i,j,k+2)+fh2(i,j,k+3)) &
          - bzn*(-F3*fh2(i,j,k+1)-F10*fh2(i,j,k)+F18*fh2(i,j,k-1)-F6*fh2(i,j,k-2)+fh2(i,j,k-3))
      f2_rhs(i,j,k)=f2_rhs(i,j,k)+ax
      f2_rhs(i,j,k)=f2_rhs(i,j,k)+ay
      f2_rhs(i,j,k)=f2_rhs(i,j,k)+az
    enddo
    enddo
    enddo
  endif
#endif

  ! Keep all boundary and low-order cases on the original implementation.
  call lopsided_core(ex,X,Y,Z,f1_rhs,Sfx,Sfy,Sfz,Symmetry,SoA1,.true.,fh1)
  call lopsided_core(ex,X,Y,Z,f2_rhs,Sfx,Sfy,Sfz,Symmetry,SoA2,.true.,fh2)
end subroutine lopsided2
