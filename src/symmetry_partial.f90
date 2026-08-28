! Partial symmetry preparation used by the lopsided ghost-only experiment.
! It copies only the positive boundary slabs needed by the shell and the
! positive source planes needed to reflect the low-side ghost cells.  The
! deep interior is read directly from the original input field.
subroutine symmetry_bd_partial(ord,extc,func,funcc,SoA,width)
  implicit none
  integer, intent(in) :: ord, width
  integer, dimension(3), intent(in) :: extc
  real*8, dimension(extc(1),extc(2),extc(3)), intent(in) :: func
  real*8, dimension(-ord+1:extc(1),-ord+1:extc(2),-ord+1:extc(3)), intent(out) :: funcc
  real*8, dimension(3), intent(in) :: SoA
  integer :: w1,w2,w3,lo1,lo2,lo3

  ! A width of ord+1 is sufficient for the order-ord ghost reflection and
  ! the one-sided shell stencil.  Small blocks fall back to a full copy so
  ! that no partially initialized region can be observed.
  w1 = min(extc(1), max(ord+1, width))
  w2 = min(extc(2), max(ord+1, width))
  w3 = min(extc(3), max(ord+1, width))
  if (extc(1) <= 2*w1 .or. extc(2) <= 2*w2 .or. extc(3) <= 2*w3) then
    funcc = 0.d0
    funcc(1:extc(1),1:extc(2),1:extc(3)) = func
  else
    ! Low and high positive slabs cover all shell points and all positive
    ! planes used as sources while constructing the low-side ghost corners.
    funcc(1:w1,1:extc(2),1:extc(3)) = func(1:w1,1:extc(2),1:extc(3))
    lo1 = extc(1)-w1+1
    funcc(lo1:extc(1),1:extc(2),1:extc(3)) = func(lo1:extc(1),1:extc(2),1:extc(3))
    funcc(1:extc(1),1:w2,1:extc(3)) = func(1:extc(1),1:w2,1:extc(3))
    lo2 = extc(2)-w2+1
    funcc(1:extc(1),lo2:extc(2),1:extc(3)) = func(1:extc(1),lo2:extc(2),1:extc(3))
    funcc(1:extc(1),1:extc(2),1:w3) = func(1:extc(1),1:extc(2),1:w3)
    lo3 = extc(3)-w3+1
    funcc(1:extc(1),1:extc(2),lo3:extc(3)) = func(1:extc(1),1:extc(2),lo3:extc(3))
  endif

  ! The order is intentional: y corners use x ghost planes and z corners
  ! use both x and y ghost planes, matching symmetry_bd exactly.
  do lo1=0,ord-1
    funcc(-lo1,1:extc(2),1:extc(3)) = funcc(lo1+1,1:extc(2),1:extc(3))*SoA(1)
  enddo
  do lo1=0,ord-1
    funcc(:,-lo1,1:extc(3)) = funcc(:,lo1+1,1:extc(3))*SoA(2)
  enddo
  do lo1=0,ord-1
    funcc(:,:,-lo1) = funcc(:,:,lo1+1)*SoA(3)
  enddo
end subroutine symmetry_bd_partial
