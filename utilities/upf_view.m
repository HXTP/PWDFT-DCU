function upf_view(ppFile, varargin)
% View the pseudopotential read from the UPF file format
%
% This requires running KSSOLV_startup first
%
% Revision: 2017/11/28

pp = upfread([], ppFile);
Znuc = pp.anum;
Zion = pp.venum;

rcps = 1.0;
rhocut =  6.0;
rhoatomcut = 4.0;
wavcut = 2.0;
if( nargin >= 2 )
  rcps  = varargin{1};
end
if( nargin >= 3 )
  rhocut = varargin{2};
end

fprintf('rcps       = %g\n', rcps);
fprintf('rhocut     = %g\n', rhocut);
fprintf('rhoatomcut = %g\n', rhoatomcut);
fprintf('wavcut     = %g\n', wavcut);

r = pp.r;

% Interpolation grid to correctly treat the value at r=0.

Vlocshort = zeros(size(pp.vloc));
Vlocshort(1) = pp.vloc(1) + Zion / rcps * (2/sqrt(pi));
Vlocshort(2:end) = pp.vloc(2:end) + Zion ./ pp.r(2:end) .* ...
  erf( pp.r(2:end) / rcps );
rhoGaussian = @(r) Zion * exp(-(r / rcps).^2) / (pi^(3/2)*rcps^3);
% Check
fprintf('Int rhoGaussian = %g\n', ...
  sum(pp.r.^2.*rhoGaussian(pp.r).*pp.rab)*4*pi);

[interpr, Vlocinterpr] = ...
  splinerad( pp.r, pp.vloc, 1 );
spVloc = csape(interpr, Vlocinterpr);
Vloc = fnval(spVloc,r);
dVloc = fnval(fnder(spVloc,1),r);

pk = find(r>0 & r<rhocut); 
fprintf('int rhoGaussian cut = %15.10e\n', ...
  sum(4*pi*r(pk).^2.*(rhoGaussian(r(pk))).*pp.rab(pk)));


figure
clf
plot(pp.r,Vlocshort,'r-.',pp.r,rhoGaussian(pp.r),'b-.');
box on
legend('VlocSR','rhoGaussian','Location','SE');
pause


% atomic core charge density. 
% Note: rhoatom read from psp8read is multiplied by 4*pi*r^2, and drhoc
% from psp8 is not used!
[interpr, rhoatominterpr] = ...
  splinerad( pp.r, pp.rhoatom, 1 );
rhoatominterpr = rhoatominterpr ./ (4*pi*interpr.^2);
sprhoatom = csape(interpr, rhoatominterpr);
rhoatom = fnval(sprhoatom,r);
drhoatom = fnval(fnder(sprhoatom,1),r);


% Model core charge density
pk = find(r>0 & r<rhoatomcut); 
fprintf('int rhoatom        = %15.10e\n', ...
  sum(4*pi*r.^2.*(rhoatom).*pp.rab));
fprintf('int rhoatom  cut   = %15.10e\n', ...
  sum(4*pi*r(pk).^2.*(rhoatom(pk)).*pp.rab(pk)));

% nonlocal pseudopotential, derivative padded by zero and not used
for l = 1 : pp.nonloc.nbeta
  if( mod(pp.nonloc.lll(l),2) == 0 )
    [interpr, nlinterpr] = ...
      splinerad(pp.r, pp.nonloc.beta(:,l),0);
  else
    [interpr, nlinterpr] = ...
      splinerad(pp.r, pp.nonloc.beta(:,l),1);
  end
  nlinterpr = nlinterpr./interpr;

  spnl = csape(interpr, nlinterpr);

  vnl = fnval( spnl, r );
  pk = find(r>0 & r<wavcut);
  sumpp = sum(r.^2.*vnl.^2.*pp.rab);
  sumppcut = sum(r(pk).^2.*vnl(pk).^2.*pp.rab(pk));
  fprintf('Abs: int p^2  = %15.10e\n', sumpp);
  fprintf('Rel: int p^2 cut = %15.10e\n', sumppcut / sumpp );

end


