! Keep each component in a compact SIMD loop while revisiting a small set of
! j rows for all six components before advancing to the next tile.
  do k=1,ex(3)
  do jb=1,ex(2),AMSS_RHS_RICCI_TILE_J
    jend=min(jb+AMSS_RHS_RICCI_TILE_J-1,ex(2))

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Rxx(i,j,k) = -HALF * Rxx(i,j,k) + &
        gxx(i,j,k)*Gamxx(i,j,k) + gxy(i,j,k)*Gamyx(i,j,k) + gxz(i,j,k)*Gamzx(i,j,k) + &
        Gamxa(i,j,k)*gxxx(i,j,k) + Gamya(i,j,k)*gxyx(i,j,k) + Gamza(i,j,k)*gxzx(i,j,k) + &
        gupxx(i,j,k)*(TWO*(Gamxxx(i,j,k)*gxxx(i,j,k) + Gamyxx(i,j,k)*gxyx(i,j,k) + Gamzxx(i,j,k)*gxzx(i,j,k)) + &
          Gamxxx(i,j,k)*gxxx(i,j,k) + Gamyxx(i,j,k)*gxxy(i,j,k) + Gamzxx(i,j,k)*gxxz(i,j,k)) + &
        gupxy(i,j,k)*(TWO*(Gamxxx(i,j,k)*gxyx(i,j,k) + Gamyxx(i,j,k)*gyyx(i,j,k) + Gamzxx(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gxxx(i,j,k) + Gamyxy(i,j,k)*gxyx(i,j,k) + Gamzxy(i,j,k)*gxzx(i,j,k)) + &
          Gamxxy(i,j,k)*gxxx(i,j,k) + Gamyxy(i,j,k)*gxxy(i,j,k) + Gamzxy(i,j,k)*gxxz(i,j,k) + &
          Gamxxx(i,j,k)*gxyx(i,j,k) + Gamyxx(i,j,k)*gxyy(i,j,k) + Gamzxx(i,j,k)*gxyz(i,j,k)) + &
        gupxz(i,j,k)*(TWO*(Gamxxx(i,j,k)*gxzx(i,j,k) + Gamyxx(i,j,k)*gyzx(i,j,k) + Gamzxx(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gxxx(i,j,k) + Gamyxz(i,j,k)*gxyx(i,j,k) + Gamzxz(i,j,k)*gxzx(i,j,k)) + &
          Gamxxz(i,j,k)*gxxx(i,j,k) + Gamyxz(i,j,k)*gxxy(i,j,k) + Gamzxz(i,j,k)*gxxz(i,j,k) + &
          Gamxxx(i,j,k)*gxzx(i,j,k) + Gamyxx(i,j,k)*gxzy(i,j,k) + Gamzxx(i,j,k)*gxzz(i,j,k)) + &
        gupyy(i,j,k)*(TWO*(Gamxxy(i,j,k)*gxyx(i,j,k) + Gamyxy(i,j,k)*gyyx(i,j,k) + Gamzxy(i,j,k)*gyzx(i,j,k)) + &
          Gamxxy(i,j,k)*gxyx(i,j,k) + Gamyxy(i,j,k)*gxyy(i,j,k) + Gamzxy(i,j,k)*gxyz(i,j,k)) + &
        gupyz(i,j,k)*(TWO*(Gamxxy(i,j,k)*gxzx(i,j,k) + Gamyxy(i,j,k)*gyzx(i,j,k) + Gamzxy(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gxyx(i,j,k) + Gamyxz(i,j,k)*gyyx(i,j,k) + Gamzxz(i,j,k)*gyzx(i,j,k)) + &
          Gamxxz(i,j,k)*gxyx(i,j,k) + Gamyxz(i,j,k)*gxyy(i,j,k) + Gamzxz(i,j,k)*gxyz(i,j,k) + &
          Gamxxy(i,j,k)*gxzx(i,j,k) + Gamyxy(i,j,k)*gxzy(i,j,k) + Gamzxy(i,j,k)*gxzz(i,j,k)) + &
        gupzz(i,j,k)*(TWO*(Gamxxz(i,j,k)*gxzx(i,j,k) + Gamyxz(i,j,k)*gyzx(i,j,k) + Gamzxz(i,j,k)*gzzx(i,j,k)) + &
          Gamxxz(i,j,k)*gxzx(i,j,k) + Gamyxz(i,j,k)*gxzy(i,j,k) + Gamzxz(i,j,k)*gxzz(i,j,k))
    enddo
    enddo

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Ryy(i,j,k) = -HALF * Ryy(i,j,k) + &
        gxy(i,j,k)*Gamxy(i,j,k) + gyy(i,j,k)*Gamyy(i,j,k) + gyz(i,j,k)*Gamzy(i,j,k) + &
        Gamxa(i,j,k)*gxyy(i,j,k) + Gamya(i,j,k)*gyyy(i,j,k) + Gamza(i,j,k)*gyzy(i,j,k) + &
        gupxx(i,j,k)*(TWO*(Gamxxy(i,j,k)*gxxy(i,j,k) + Gamyxy(i,j,k)*gxyy(i,j,k) + Gamzxy(i,j,k)*gxzy(i,j,k)) + &
          Gamxxy(i,j,k)*gxyx(i,j,k) + Gamyxy(i,j,k)*gxyy(i,j,k) + Gamzxy(i,j,k)*gxyz(i,j,k)) + &
        gupxy(i,j,k)*(TWO*(Gamxxy(i,j,k)*gxyy(i,j,k) + Gamyxy(i,j,k)*gyyy(i,j,k) + Gamzxy(i,j,k)*gyzy(i,j,k) + &
          Gamxyy(i,j,k)*gxxy(i,j,k) + Gamyyy(i,j,k)*gxyy(i,j,k) + Gamzyy(i,j,k)*gxzy(i,j,k)) + &
          Gamxyy(i,j,k)*gxyx(i,j,k) + Gamyyy(i,j,k)*gxyy(i,j,k) + Gamzyy(i,j,k)*gxyz(i,j,k) + &
          Gamxxy(i,j,k)*gyyx(i,j,k) + Gamyxy(i,j,k)*gyyy(i,j,k) + Gamzxy(i,j,k)*gyyz(i,j,k)) + &
        gupxz(i,j,k)*(TWO*(Gamxxy(i,j,k)*gxzy(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gxxy(i,j,k) + Gamyyz(i,j,k)*gxyy(i,j,k) + Gamzyz(i,j,k)*gxzy(i,j,k)) + &
          Gamxyz(i,j,k)*gxyx(i,j,k) + Gamyyz(i,j,k)*gxyy(i,j,k) + Gamzyz(i,j,k)*gxyz(i,j,k) + &
          Gamxxy(i,j,k)*gyzx(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k)) + &
        gupyy(i,j,k)*(TWO*(Gamxyy(i,j,k)*gxyy(i,j,k) + Gamyyy(i,j,k)*gyyy(i,j,k) + Gamzyy(i,j,k)*gyzy(i,j,k)) + &
          Gamxyy(i,j,k)*gyyx(i,j,k) + Gamyyy(i,j,k)*gyyy(i,j,k) + Gamzyy(i,j,k)*gyyz(i,j,k)) + &
        gupyz(i,j,k)*(TWO*(Gamxyy(i,j,k)*gxzy(i,j,k) + Gamyyy(i,j,k)*gyzy(i,j,k) + Gamzyy(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gxyy(i,j,k) + Gamyyz(i,j,k)*gyyy(i,j,k) + Gamzyz(i,j,k)*gyzy(i,j,k)) + &
          Gamxyz(i,j,k)*gyyx(i,j,k) + Gamyyz(i,j,k)*gyyy(i,j,k) + Gamzyz(i,j,k)*gyyz(i,j,k) + &
          Gamxyy(i,j,k)*gyzx(i,j,k) + Gamyyy(i,j,k)*gyzy(i,j,k) + Gamzyy(i,j,k)*gyzz(i,j,k)) + &
        gupzz(i,j,k)*(TWO*(Gamxyz(i,j,k)*gxzy(i,j,k) + Gamyyz(i,j,k)*gyzy(i,j,k) + Gamzyz(i,j,k)*gzzy(i,j,k)) + &
          Gamxyz(i,j,k)*gyzx(i,j,k) + Gamyyz(i,j,k)*gyzy(i,j,k) + Gamzyz(i,j,k)*gyzz(i,j,k))
    enddo
    enddo

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Rzz(i,j,k) = -HALF * Rzz(i,j,k) + &
        gxz(i,j,k)*Gamxz(i,j,k) + gyz(i,j,k)*Gamyz(i,j,k) + gzz(i,j,k)*Gamzz(i,j,k) + &
        Gamxa(i,j,k)*gxzz(i,j,k) + Gamya(i,j,k)*gyzz(i,j,k) + Gamza(i,j,k)*gzzz(i,j,k) + &
        gupxx(i,j,k)*(TWO*(Gamxxz(i,j,k)*gxxz(i,j,k) + Gamyxz(i,j,k)*gxyz(i,j,k) + Gamzxz(i,j,k)*gxzz(i,j,k)) + &
          Gamxxz(i,j,k)*gxzx(i,j,k) + Gamyxz(i,j,k)*gxzy(i,j,k) + Gamzxz(i,j,k)*gxzz(i,j,k)) + &
        gupxy(i,j,k)*(TWO*(Gamxxz(i,j,k)*gxyz(i,j,k) + Gamyxz(i,j,k)*gyyz(i,j,k) + Gamzxz(i,j,k)*gyzz(i,j,k) + &
          Gamxyz(i,j,k)*gxxz(i,j,k) + Gamyyz(i,j,k)*gxyz(i,j,k) + Gamzyz(i,j,k)*gxzz(i,j,k)) + &
          Gamxyz(i,j,k)*gxzx(i,j,k) + Gamyyz(i,j,k)*gxzy(i,j,k) + Gamzyz(i,j,k)*gxzz(i,j,k) + &
          Gamxxz(i,j,k)*gyzx(i,j,k) + Gamyxz(i,j,k)*gyzy(i,j,k) + Gamzxz(i,j,k)*gyzz(i,j,k)) + &
        gupxz(i,j,k)*(TWO*(Gamxxz(i,j,k)*gxzz(i,j,k) + Gamyxz(i,j,k)*gyzz(i,j,k) + Gamzxz(i,j,k)*gzzz(i,j,k) + &
          Gamxzz(i,j,k)*gxxz(i,j,k) + Gamyzz(i,j,k)*gxyz(i,j,k) + Gamzzz(i,j,k)*gxzz(i,j,k)) + &
          Gamxzz(i,j,k)*gxzx(i,j,k) + Gamyzz(i,j,k)*gxzy(i,j,k) + Gamzzz(i,j,k)*gxzz(i,j,k) + &
          Gamxxz(i,j,k)*gzzx(i,j,k) + Gamyxz(i,j,k)*gzzy(i,j,k) + Gamzxz(i,j,k)*gzzz(i,j,k)) + &
        gupyy(i,j,k)*(TWO*(Gamxyz(i,j,k)*gxyz(i,j,k) + Gamyyz(i,j,k)*gyyz(i,j,k) + Gamzyz(i,j,k)*gyzz(i,j,k)) + &
          Gamxyz(i,j,k)*gyzx(i,j,k) + Gamyyz(i,j,k)*gyzy(i,j,k) + Gamzyz(i,j,k)*gyzz(i,j,k)) + &
        gupyz(i,j,k)*(TWO*(Gamxyz(i,j,k)*gxzz(i,j,k) + Gamyyz(i,j,k)*gyzz(i,j,k) + Gamzyz(i,j,k)*gzzz(i,j,k) + &
          Gamxzz(i,j,k)*gxyz(i,j,k) + Gamyzz(i,j,k)*gyyz(i,j,k) + Gamzzz(i,j,k)*gyzz(i,j,k)) + &
          Gamxzz(i,j,k)*gyzx(i,j,k) + Gamyzz(i,j,k)*gyzy(i,j,k) + Gamzzz(i,j,k)*gyzz(i,j,k) + &
          Gamxyz(i,j,k)*gzzx(i,j,k) + Gamyyz(i,j,k)*gzzy(i,j,k) + Gamzyz(i,j,k)*gzzz(i,j,k)) + &
        gupzz(i,j,k)*(TWO*(Gamxzz(i,j,k)*gxzz(i,j,k) + Gamyzz(i,j,k)*gyzz(i,j,k) + Gamzzz(i,j,k)*gzzz(i,j,k)) + &
          Gamxzz(i,j,k)*gzzx(i,j,k) + Gamyzz(i,j,k)*gzzy(i,j,k) + Gamzzz(i,j,k)*gzzz(i,j,k))
    enddo
    enddo

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Rxy(i,j,k) = HALF*(-Rxy(i,j,k) + &
        gxx(i,j,k)*Gamxy(i,j,k) + gxy(i,j,k)*Gamyy(i,j,k) + gxz(i,j,k)*Gamzy(i,j,k) + &
        gxy(i,j,k)*Gamxx(i,j,k) + gyy(i,j,k)*Gamyx(i,j,k) + gyz(i,j,k)*Gamzx(i,j,k) + &
        Gamxa(i,j,k)*gxyx(i,j,k) + Gamya(i,j,k)*gyyx(i,j,k) + Gamza(i,j,k)*gyzx(i,j,k) + &
        Gamxa(i,j,k)*gxxy(i,j,k) + Gamya(i,j,k)*gxyy(i,j,k) + Gamza(i,j,k)*gxzy(i,j,k)) + &
        gupxx(i,j,k)*(Gamxxx(i,j,k)*gxxy(i,j,k) + Gamyxx(i,j,k)*gxyy(i,j,k) + Gamzxx(i,j,k)*gxzy(i,j,k) + &
          Gamxxy(i,j,k)*gxxx(i,j,k) + Gamyxy(i,j,k)*gxyx(i,j,k) + Gamzxy(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gxyx(i,j,k) + Gamyxx(i,j,k)*gxyy(i,j,k) + Gamzxx(i,j,k)*gxyz(i,j,k)) + &
        gupxy(i,j,k)*(Gamxxx(i,j,k)*gxyy(i,j,k) + Gamyxx(i,j,k)*gyyy(i,j,k) + Gamzxx(i,j,k)*gyzy(i,j,k) + &
          Gamxxy(i,j,k)*gxyx(i,j,k) + Gamyxy(i,j,k)*gyyx(i,j,k) + Gamzxy(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gxyx(i,j,k) + Gamyxy(i,j,k)*gxyy(i,j,k) + Gamzxy(i,j,k)*gxyz(i,j,k) + &
          Gamxxy(i,j,k)*gxxy(i,j,k) + Gamyxy(i,j,k)*gxyy(i,j,k) + Gamzxy(i,j,k)*gxzy(i,j,k) + &
          Gamxyy(i,j,k)*gxxx(i,j,k) + Gamyyy(i,j,k)*gxyx(i,j,k) + Gamzyy(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gyyx(i,j,k) + Gamyxx(i,j,k)*gyyy(i,j,k) + Gamzxx(i,j,k)*gyyz(i,j,k)) + &
        gupxz(i,j,k)*(Gamxxx(i,j,k)*gxzy(i,j,k) + Gamyxx(i,j,k)*gyzy(i,j,k) + Gamzxx(i,j,k)*gzzy(i,j,k) + &
          Gamxxy(i,j,k)*gxzx(i,j,k) + Gamyxy(i,j,k)*gyzx(i,j,k) + Gamzxy(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gxyx(i,j,k) + Gamyxz(i,j,k)*gxyy(i,j,k) + Gamzxz(i,j,k)*gxyz(i,j,k) + &
          Gamxxz(i,j,k)*gxxy(i,j,k) + Gamyxz(i,j,k)*gxyy(i,j,k) + Gamzxz(i,j,k)*gxzy(i,j,k) + &
          Gamxyz(i,j,k)*gxxx(i,j,k) + Gamyyz(i,j,k)*gxyx(i,j,k) + Gamzyz(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gyzx(i,j,k) + Gamyxx(i,j,k)*gyzy(i,j,k) + Gamzxx(i,j,k)*gyzz(i,j,k)) + &
        gupyy(i,j,k)*(Gamxxy(i,j,k)*gxyy(i,j,k) + Gamyxy(i,j,k)*gyyy(i,j,k) + Gamzxy(i,j,k)*gyzy(i,j,k) + &
          Gamxyy(i,j,k)*gxyx(i,j,k) + Gamyyy(i,j,k)*gyyx(i,j,k) + Gamzyy(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gyyx(i,j,k) + Gamyxy(i,j,k)*gyyy(i,j,k) + Gamzxy(i,j,k)*gyyz(i,j,k)) + &
        gupyz(i,j,k)*(Gamxxy(i,j,k)*gxzy(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gzzy(i,j,k) + &
          Gamxyy(i,j,k)*gxzx(i,j,k) + Gamyyy(i,j,k)*gyzx(i,j,k) + Gamzyy(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gyyx(i,j,k) + Gamyxz(i,j,k)*gyyy(i,j,k) + Gamzxz(i,j,k)*gyyz(i,j,k) + &
          Gamxxz(i,j,k)*gxyy(i,j,k) + Gamyxz(i,j,k)*gyyy(i,j,k) + Gamzxz(i,j,k)*gyzy(i,j,k) + &
          Gamxyz(i,j,k)*gxyx(i,j,k) + Gamyyz(i,j,k)*gyyx(i,j,k) + Gamzyz(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gyzx(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k)) + &
        gupzz(i,j,k)*(Gamxxz(i,j,k)*gxzy(i,j,k) + Gamyxz(i,j,k)*gyzy(i,j,k) + Gamzxz(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gxzx(i,j,k) + Gamyyz(i,j,k)*gyzx(i,j,k) + Gamzyz(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gyzx(i,j,k) + Gamyxz(i,j,k)*gyzy(i,j,k) + Gamzxz(i,j,k)*gyzz(i,j,k))
    enddo
    enddo

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Rxz(i,j,k) = HALF*(-Rxz(i,j,k) + &
        gxx(i,j,k)*Gamxz(i,j,k) + gxy(i,j,k)*Gamyz(i,j,k) + gxz(i,j,k)*Gamzz(i,j,k) + &
        gxz(i,j,k)*Gamxx(i,j,k) + gyz(i,j,k)*Gamyx(i,j,k) + gzz(i,j,k)*Gamzx(i,j,k) + &
        Gamxa(i,j,k)*gxzx(i,j,k) + Gamya(i,j,k)*gyzx(i,j,k) + Gamza(i,j,k)*gzzx(i,j,k) + &
        Gamxa(i,j,k)*gxxz(i,j,k) + Gamya(i,j,k)*gxyz(i,j,k) + Gamza(i,j,k)*gxzz(i,j,k)) + &
        gupxx(i,j,k)*(Gamxxx(i,j,k)*gxxz(i,j,k) + Gamyxx(i,j,k)*gxyz(i,j,k) + Gamzxx(i,j,k)*gxzz(i,j,k) + &
          Gamxxz(i,j,k)*gxxx(i,j,k) + Gamyxz(i,j,k)*gxyx(i,j,k) + Gamzxz(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gxzx(i,j,k) + Gamyxx(i,j,k)*gxzy(i,j,k) + Gamzxx(i,j,k)*gxzz(i,j,k)) + &
        gupxy(i,j,k)*(Gamxxx(i,j,k)*gxyz(i,j,k) + Gamyxx(i,j,k)*gyyz(i,j,k) + Gamzxx(i,j,k)*gyzz(i,j,k) + &
          Gamxxz(i,j,k)*gxyx(i,j,k) + Gamyxz(i,j,k)*gyyx(i,j,k) + Gamzxz(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gxzx(i,j,k) + Gamyxy(i,j,k)*gxzy(i,j,k) + Gamzxy(i,j,k)*gxzz(i,j,k) + &
          Gamxxy(i,j,k)*gxxz(i,j,k) + Gamyxy(i,j,k)*gxyz(i,j,k) + Gamzxy(i,j,k)*gxzz(i,j,k) + &
          Gamxyz(i,j,k)*gxxx(i,j,k) + Gamyyz(i,j,k)*gxyx(i,j,k) + Gamzyz(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gyzx(i,j,k) + Gamyxx(i,j,k)*gyzy(i,j,k) + Gamzxx(i,j,k)*gyzz(i,j,k)) + &
        gupxz(i,j,k)*(Gamxxx(i,j,k)*gxzz(i,j,k) + Gamyxx(i,j,k)*gyzz(i,j,k) + Gamzxx(i,j,k)*gzzz(i,j,k) + &
          Gamxxz(i,j,k)*gxzx(i,j,k) + Gamyxz(i,j,k)*gyzx(i,j,k) + Gamzxz(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gxzx(i,j,k) + Gamyxz(i,j,k)*gxzy(i,j,k) + Gamzxz(i,j,k)*gxzz(i,j,k) + &
          Gamxxz(i,j,k)*gxxz(i,j,k) + Gamyxz(i,j,k)*gxyz(i,j,k) + Gamzxz(i,j,k)*gxzz(i,j,k) + &
          Gamxzz(i,j,k)*gxxx(i,j,k) + Gamyzz(i,j,k)*gxyx(i,j,k) + Gamzzz(i,j,k)*gxzx(i,j,k) + &
          Gamxxx(i,j,k)*gzzx(i,j,k) + Gamyxx(i,j,k)*gzzy(i,j,k) + Gamzxx(i,j,k)*gzzz(i,j,k)) + &
        gupyy(i,j,k)*(Gamxxy(i,j,k)*gxyz(i,j,k) + Gamyxy(i,j,k)*gyyz(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k) + &
          Gamxyz(i,j,k)*gxyx(i,j,k) + Gamyyz(i,j,k)*gyyx(i,j,k) + Gamzyz(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gyzx(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k)) + &
        gupyz(i,j,k)*(Gamxxy(i,j,k)*gxzz(i,j,k) + Gamyxy(i,j,k)*gyzz(i,j,k) + Gamzxy(i,j,k)*gzzz(i,j,k) + &
          Gamxyz(i,j,k)*gxzx(i,j,k) + Gamyyz(i,j,k)*gyzx(i,j,k) + Gamzyz(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gyzx(i,j,k) + Gamyxz(i,j,k)*gyzy(i,j,k) + Gamzxz(i,j,k)*gyzz(i,j,k) + &
          Gamxxz(i,j,k)*gxyz(i,j,k) + Gamyxz(i,j,k)*gyyz(i,j,k) + Gamzxz(i,j,k)*gyzz(i,j,k) + &
          Gamxzz(i,j,k)*gxyx(i,j,k) + Gamyzz(i,j,k)*gyyx(i,j,k) + Gamzzz(i,j,k)*gyzx(i,j,k) + &
          Gamxxy(i,j,k)*gzzx(i,j,k) + Gamyxy(i,j,k)*gzzy(i,j,k) + Gamzxy(i,j,k)*gzzz(i,j,k)) + &
        gupzz(i,j,k)*(Gamxxz(i,j,k)*gxzz(i,j,k) + Gamyxz(i,j,k)*gyzz(i,j,k) + Gamzxz(i,j,k)*gzzz(i,j,k) + &
          Gamxzz(i,j,k)*gxzx(i,j,k) + Gamyzz(i,j,k)*gyzx(i,j,k) + Gamzzz(i,j,k)*gzzx(i,j,k) + &
          Gamxxz(i,j,k)*gzzx(i,j,k) + Gamyxz(i,j,k)*gzzy(i,j,k) + Gamzxz(i,j,k)*gzzz(i,j,k))
    enddo
    enddo

    do j=jb,jend
!$omp simd
    do i=1,ex(1)
      Ryz(i,j,k) = HALF*(-Ryz(i,j,k) + &
        gxy(i,j,k)*Gamxz(i,j,k) + gyy(i,j,k)*Gamyz(i,j,k) + gyz(i,j,k)*Gamzz(i,j,k) + &
        gxz(i,j,k)*Gamxy(i,j,k) + gyz(i,j,k)*Gamyy(i,j,k) + gzz(i,j,k)*Gamzy(i,j,k) + &
        Gamxa(i,j,k)*gxzy(i,j,k) + Gamya(i,j,k)*gyzy(i,j,k) + Gamza(i,j,k)*gzzy(i,j,k) + &
        Gamxa(i,j,k)*gxyz(i,j,k) + Gamya(i,j,k)*gyyz(i,j,k) + Gamza(i,j,k)*gyzz(i,j,k)) + &
        gupxx(i,j,k)*(Gamxxy(i,j,k)*gxxz(i,j,k) + Gamyxy(i,j,k)*gxyz(i,j,k) + Gamzxy(i,j,k)*gxzz(i,j,k) + &
          Gamxxz(i,j,k)*gxxy(i,j,k) + Gamyxz(i,j,k)*gxyy(i,j,k) + Gamzxz(i,j,k)*gxzy(i,j,k) + &
          Gamxxy(i,j,k)*gxzx(i,j,k) + Gamyxy(i,j,k)*gxzy(i,j,k) + Gamzxy(i,j,k)*gxzz(i,j,k)) + &
        gupxy(i,j,k)*(Gamxxy(i,j,k)*gxyz(i,j,k) + Gamyxy(i,j,k)*gyyz(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k) + &
          Gamxxz(i,j,k)*gxyy(i,j,k) + Gamyxz(i,j,k)*gyyy(i,j,k) + Gamzxz(i,j,k)*gyzy(i,j,k) + &
          Gamxyy(i,j,k)*gxzx(i,j,k) + Gamyyy(i,j,k)*gxzy(i,j,k) + Gamzyy(i,j,k)*gxzz(i,j,k) + &
          Gamxyy(i,j,k)*gxxz(i,j,k) + Gamyyy(i,j,k)*gxyz(i,j,k) + Gamzyy(i,j,k)*gxzz(i,j,k) + &
          Gamxyz(i,j,k)*gxxy(i,j,k) + Gamyyz(i,j,k)*gxyy(i,j,k) + Gamzyz(i,j,k)*gxzy(i,j,k) + &
          Gamxxy(i,j,k)*gyzx(i,j,k) + Gamyxy(i,j,k)*gyzy(i,j,k) + Gamzxy(i,j,k)*gyzz(i,j,k)) + &
        gupxz(i,j,k)*(Gamxxy(i,j,k)*gxzz(i,j,k) + Gamyxy(i,j,k)*gyzz(i,j,k) + Gamzxy(i,j,k)*gzzz(i,j,k) + &
          Gamxxz(i,j,k)*gxzy(i,j,k) + Gamyxz(i,j,k)*gyzy(i,j,k) + Gamzxz(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gxzx(i,j,k) + Gamyyz(i,j,k)*gxzy(i,j,k) + Gamzyz(i,j,k)*gxzz(i,j,k) + &
          Gamxyz(i,j,k)*gxxz(i,j,k) + Gamyyz(i,j,k)*gxyz(i,j,k) + Gamzyz(i,j,k)*gxzz(i,j,k) + &
          Gamxzz(i,j,k)*gxxy(i,j,k) + Gamyzz(i,j,k)*gxyy(i,j,k) + Gamzzz(i,j,k)*gxzy(i,j,k) + &
          Gamxxy(i,j,k)*gzzx(i,j,k) + Gamyxy(i,j,k)*gzzy(i,j,k) + Gamzxy(i,j,k)*gzzz(i,j,k)) + &
        gupyy(i,j,k)*(Gamxyy(i,j,k)*gxyz(i,j,k) + Gamyyy(i,j,k)*gyyz(i,j,k) + Gamzyy(i,j,k)*gyzz(i,j,k) + &
          Gamxyz(i,j,k)*gxyy(i,j,k) + Gamyyz(i,j,k)*gyyy(i,j,k) + Gamzyz(i,j,k)*gyzy(i,j,k) + &
          Gamxyy(i,j,k)*gyzx(i,j,k) + Gamyyy(i,j,k)*gyzy(i,j,k) + Gamzyy(i,j,k)*gyzz(i,j,k)) + &
        gupyz(i,j,k)*(Gamxyy(i,j,k)*gxzz(i,j,k) + Gamyyy(i,j,k)*gyzz(i,j,k) + Gamzyy(i,j,k)*gzzz(i,j,k) + &
          Gamxyz(i,j,k)*gxzy(i,j,k) + Gamyyz(i,j,k)*gyzy(i,j,k) + Gamzyz(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gyzx(i,j,k) + Gamyyz(i,j,k)*gyzy(i,j,k) + Gamzyz(i,j,k)*gyzz(i,j,k) + &
          Gamxyz(i,j,k)*gxyz(i,j,k) + Gamyyz(i,j,k)*gyyz(i,j,k) + Gamzyz(i,j,k)*gyzz(i,j,k) + &
          Gamxzz(i,j,k)*gxyy(i,j,k) + Gamyzz(i,j,k)*gyyy(i,j,k) + Gamzzz(i,j,k)*gyzy(i,j,k) + &
          Gamxyy(i,j,k)*gzzx(i,j,k) + Gamyyy(i,j,k)*gzzy(i,j,k) + Gamzyy(i,j,k)*gzzz(i,j,k)) + &
        gupzz(i,j,k)*(Gamxyz(i,j,k)*gxzz(i,j,k) + Gamyyz(i,j,k)*gyzz(i,j,k) + Gamzyz(i,j,k)*gzzz(i,j,k) + &
          Gamxzz(i,j,k)*gxzy(i,j,k) + Gamyzz(i,j,k)*gyzy(i,j,k) + Gamzzz(i,j,k)*gzzy(i,j,k) + &
          Gamxyz(i,j,k)*gzzx(i,j,k) + Gamyyz(i,j,k)*gzzy(i,j,k) + Gamzyz(i,j,k)*gzzz(i,j,k))
    enddo
    enddo
  enddo
  enddo
