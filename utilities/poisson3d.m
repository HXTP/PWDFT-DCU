% POISSON3D solves a model Poisson problem in 3D.
% This is used to test the Fourier routines.
% 
% Lin Lin
% 1/9/2012

Ls = [1 1 1];
Ns = [32 32 32];
x = (0:Ns(1)-1)' / Ns(1) * Ls(1);
y = (0:Ns(2)-1)' / Ns(2) * Ls(2);
z = (0:Ns(3)-1)' / Ns(3) * Ls(3);
kx = [0:Ns(1)/2 -Ns(1)/2+1:-1]' * 2 * pi / Ls(1);
ky = [0:Ns(2)/2 -Ns(2)/2+1:-1]' * 2 * pi / Ls(2);
kz = [0:Ns(3)/2 -Ns(3)/2+1:-1]' * 2 * pi / Ls(3);

[XX,YY,ZZ] = ndgrid( x, y, z );
[KXX,KYY,KZZ] = ndgrid( kx, ky, kz );

sigma = 0.2;
rho = exp(-((XX-0.5).^2 + (YY-0.5).^2 + (ZZ-0.5).^2) / (2*sigma^2));
GKK = 1/2 * (KXX.^2 + KYY.^2 + KZZ.^2);

tt = fftn(rho);
tt = tt ./ GKK * 2 * pi;
tt(find(GKK==0)) = 0;

vhart = real(ifftn(tt));
slice(YY,XX,ZZ,vhart,[], [0.1 0.5 0.8], [])
