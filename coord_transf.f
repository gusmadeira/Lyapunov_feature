      subroutine coord_h2b(nbod,mass,xh,yh,zh,vxh,vyh,vzh,
     &     xb,yb,zb,vxb,vyb,vzb,msys, Rxc,Ryc,Rzc,Vxc,Vyc,Vzc)
      implicit none
      integer nbod,i
      real*8 mass(*),xh(*),yh(*),zh(*),vxh(*),vyh(*),vzh(*)
      real*8 xb(*),yb(*),zb(*),vxb(*),vyb(*),vzb(*)
      real*8 msys, M
      real*8 Rxc,Ryc,Rzc,Vxc,Vyc,Vzc
      M = 0d0
      Rxc = 0d0
      Ryc = 0d0
      Rzc = 0d0
      Vxc = 0d0
      Vyc = 0d0
      Vzc = 0d0
      do i=1,nbod
         M  = M  + mass(i)
         Rxc = Rxc + mass(i)*xh(i)
         Ryc = Ryc + mass(i)*yh(i)
         Rzc = Rzc + mass(i)*zh(i)
         Vxc = Vxc + mass(i)*vxh(i)
         Vyc = Vyc + mass(i)*vyh(i)
         Vzc = Vzc + mass(i)*vzh(i)
      enddo
      Rxc = Rxc/M
      Ryc = Ryc/M
      Rzc = Rzc/M
      Vxc = Vxc/M
      Vyc = Vyc/M
      Vzc = Vzc/M
      do i=1,nbod
         xb(i)  = xh(i)  - Rxc
         yb(i)  = yh(i)  - Ryc
         zb(i)  = zh(i)  - Rzc
         vxb(i) = vxh(i) - Vxc
         vyb(i) = vyh(i) - Vyc
         vzb(i) = vzh(i) - Vzc
      enddo
      msys = M
      return
      end

      subroutine coord_h2b_tp(ntp, xht, yht, zht, vxht, vyht, vzht,
     &     Rxc, Ryc, Rzc, Vxc, Vyc, Vzc,
     &     xbt, ybt, zbt, vxbt, vybt, vzbt)
      implicit none
      integer ntp, i
      real*8 xht(*),yht(*),zht(*),vxht(*),vyht(*),vzht(*)
      real*8 xbt(*),ybt(*),zbt(*),vxbt(*),vybt(*),vzbt(*)
      real*8 Rxc, Ryc, Rzc, Vxc, Vyc, Vzc

      do i=1, ntp
         xbt(i)  = xht(i)  - Rxc
         ybt(i)  = yht(i)  - Ryc
         zbt(i)  = zht(i)  - Rzc
         vxbt(i) = vxht(i) - Vxc
         vybt(i) = vyht(i) - Vyc
         vzbt(i) = vzht(i) - Vzc
      enddo

      return
      end


      subroutine coord_b2h(nbod,mass,xb,yb,zb,vxb,vyb,vzb,
     &         xh,yh,zh,vxh,vyh,vzh)
      implicit none
      integer nbod,i
      real*8 mass(*),xb(*),yb(*),zb(*),vxb(*),vyb(*),vzb(*)
      real*8 xh(*),yh(*),zh(*),vxh(*),vyh(*),vzh(*)

      do i = 1, nbod
         xh(i)  = xb(i)  - xb(1)
         yh(i)  = yb(i)  - yb(1)
         zh(i)  = zb(i)  - zb(1)
         vxh(i) = vxb(i) - vxb(1)
         vyh(i) = vyb(i) - vyb(1)
         vzh(i) = vzb(i) - vzb(1)
      enddo
      return
      end

      subroutine coord_b2h_tp(ntp,xbt,ybt,zbt,vxbt,vybt,vzbt,
     &         xb,yb,zb,vxb,vyb,vzb,
     &         xht,yht,zht,vxht,vyht,vzht)
      implicit none
      integer ntp,i
      real*8 xbt(*),ybt(*),zbt(*),vxbt(*),vybt(*),vzbt(*)
      real*8 xb(*),yb(*),zb(*),vxb(*),vyb(*),vzb(*)
      real*8 xht(*),yht(*),zht(*),vxht(*),vyht(*),vzht(*)

      do i = 1, ntp
         xht(i)  = xbt(i)  - xb(1)
         yht(i)  = ybt(i)  - yb(1)
         zht(i)  = zbt(i)  - zb(1)

         vxht(i) = vxbt(i) - vxb(1)
         vyht(i) = vybt(i) - vyb(1)
         vzht(i) = vzbt(i) - vzb(1)
      enddo
      return
      end

      subroutine ine2rot(t,Omega, 
     &         xI, yI, zI, vxI, vyI, vzI,
     &         xR, yR, zR, vxR, vyR, vzR)

      implicit none
      real*8 t, Omega
      real*8 xI, yI, zI, vxI, vyI, vzI
      real*8 xR, yR, zR, vxR, vyR, vzR
      real*8 theta, cth, sth

      theta = Omega * t
      cth   = dcos(theta)
      sth   = dsin(theta)

      xR =  cth*xI + sth*yI
      yR = -sth*xI + cth*yI
      zR =  zI

      vxR =  cth*vxI + sth*vyI + Omega*yR
      vyR = -sth*vxI + cth*vyI - Omega*xR
      vzR =  vzI

      return
      end

      subroutine aei2xv(m,m0,
     &   a,e,inc,g,omega,Me,x,y,z)

      implicit none

      real*8 m,a,e,inc,g,omega,Me,m0
      real*8 x,y,z, tol, n, sq1pe
      real*8 Er, E0, ERR
      real*8 sq1me,sE2,cE2,f,r
      real*8 xpf,ypf,zpf
      real*8 x1,y1,z1,x2,y2,z2,x3,y3,z3

      n = sqrt((m + m0)/(a*a*a))

      tol = 1.0d-12

      Er   = Me
      ERR = dabs(Me)
 10   continue
      E0 = Er
      Er  = E0 - (E0 - e*dsin(E0) - Me) / (1.d0 - e*dcos(E0))
      ERR = dabs(Er - E0)
      if (ERR .gt. tol) goto 10

      sq1pe = sqrt(1.d0 + e)
      sq1me = sqrt(1.d0 - e)
      sE2   = sin(0.5d0*Er)
      cE2   = cos(0.5d0*Er)
      f     = 2.d0 * atan2(sq1pe*sE2,sq1me*cE2)

      r = (a*(1.d0 - e*e)) / (1.d0 + e*cos(f))

      xpf = r*cos(f)
      ypf = r*sin(f)
      zpf = 0.d0

      x1 =  cos(g)*xpf - sin(g)*ypf
      y1 =  sin(g)*xpf + cos(g)*ypf
      z1 =  zpf

      x2 = x1
      y2 = cos(inc)*y1 - sin(inc)*z1
      z2 = sin(inc)*y1 + cos(inc)*z1

      x3 =  cos(omega)*x2 - sin(omega)*y2
      y3 =  sin(omega)*x2 + cos(omega)*y2
      z3 =  z2

      x = x3
      y = y3
      z = z3

      return
      end

c==============================================================
c  Rotation builder: inertial -> frame with Z = pole direction
c==============================================================
      subroutine rot_I_to_spinaxis(lam, bet, RLI)

      implicit none

      real*8 lam, bet, RLI(3,3)
      real*8 zLx,zLy,zLz, xLx,xLy,xLz, yLx,yLy,yLz
      real*8 kx,ky,kz, nx,ny,nz, norm

      kx=0.d0; ky=0.d0; kz=1.d0

      zLx = cos(bet)*cos(lam)
      zLy = cos(bet)*sin(lam)
      zLz = sin(bet)

      nx = ky*zLz - kz*zLy
      ny = kz*zLx - kx*zLz
      nz = kx*zLy - ky*zLx
      norm = sqrt(nx*nx + ny*ny + nz*nz)
      xLx = nx/norm
      xLy = ny/norm
      xLz = nz/norm

      yLx = zLy*xLz - zLz*xLy
      yLy = zLz*xLx - zLx*xLz
      yLz = zLx*xLy - zLy*xLx

      RLI(1,1)=xLx; RLI(1,2)=xLy; RLI(1,3)=xLz
      RLI(2,1)=yLx; RLI(2,2)=yLy; RLI(2,3)=yLz
      RLI(3,1)=zLx; RLI(3,2)=zLy; RLI(3,3)=zLz

      return
      end

c==============================================================
c  Matrix-vector multiply (3x3)
c==============================================================
      subroutine matvec3(A, x, y, z, xo, yo, zo)

      implicit none

      real*8 A(3,3), x,y,z, xo,yo,zo

      xo = A(1,1)*x + A(1,2)*y + A(1,3)*z
      yo = A(2,1)*x + A(2,2)*y + A(2,3)*z
      zo = A(3,1)*x + A(3,2)*y + A(3,3)*z

      return
      end