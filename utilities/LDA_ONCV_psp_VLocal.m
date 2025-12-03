% Generate pseudopotential from psp8 file format to the binary format
% that can be read by PWDFT/DGDFT (2016 or later).
% 
% Revision: 2016/11/06
% Revision: 2016/11/22 Extend the nonlocal pseudopotential as an
% odd function first and then interpolate
% Revision: 2016/11/25 Use the better interpolation scheme for local
% charge density, model core density and their derivatives.
% Revision: 2017/02/01 Incorporate with new LDA pseudopotential

% Znucs = [1 3 6 8 9 15];
Znucs = [1 8];
res = cell(length(Znucs),2);
close all;
% FIXME hard coded from qbox for the radius of the Gaussian charge


for g=1:length(Znucs)
  Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

  % NOTE: rhoatomcut is only chosen to achieve around 95% of the charge! 
  % For large systems it is expensive to generate the initial density in
  % the real space, and Fourier space with structure factor could be a
  % better idea.
  switch(Znuc) case 1 % H
      mass = 1.00794;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 3.5;
      wavcut  = 1.5;
      ppFile = './ONCV_Pask/20170201_LDA/H/H.1.02.oncvpsp.psp8';
    case 3 % Li
      mass = 6.941;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 6.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/Li/Li.1.02.oncvpsp.psp8';
    case 6 % C
      mass = 12.011;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/C/C.1.02.oncvpsp.psp8';
    case 8 % O
      mass = 15.9994;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      % This pseudopotential seems to introduce ghost states!
      % ppFile = './ONCV_Pask/20170201_LDA/O/O.1.02.oncvpsp.psp8';
      
      % The ghost state at least disappears in water molecule.
      ppFile = './ONCV_Pask/abinit/O.psp8';
    case 9 % F
      mass = 18.9984032;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/F/F.1.02.oncvpsp.psp8';
    case 13 % Al
      mass = 26.9815386;
      rcps = 1.2;
      rhocut  = 6.0;
      rhoatomcut = 5.0;
      wavcut  = 2.0;
      % ppFile = './ONCV_Pask/20170201_LDA/Al/Al.1.02.oncvpsp.psp8';
      ppFile = './ONCV_Pask/abinit/Al.psp8';
    case 15 % P
      mass = 30.973762;
      rcps = 1.0;
      rhocut  = 6.0;
      rhoatomcut = 5.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/P/P.1.02.oncvpsp.psp8';
  end

  pp = psp8read( [], ppFile );
  if( Znuc ~= pp.anum )
    error('Znuc != pp.anum, reading the wrong pseudopotential file');
  end
  Zion = pp.venum;
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
  

  % atomic core charge density. 
  % Note: rhoatom read from psp8read is multiplied by 4*pi*r^2, and drhoc
  % from psp8 is not used!
  [interpr, rhoatominterpr] = ...
    splinerad( pp.r, pp.rhoatom, 1 );
  rhoatominterpr = rhoatominterpr ./ (4*pi*interpr.^2);
  sprhoatom = csape(interpr, rhoatominterpr);
  rhoatom = fnval(sprhoatom,r);
  drhoatom = fnval(fnder(sprhoatom,1),r);


  % Instead of storing the self energy, the 4-th entry is the Gaussian
  % pseudocharge
  % cur = [Znuc mass Zion Es];
  cur = [Znuc mass Zion rcps];

  spl = zeros(numel(r), 0);
  wgt = zeros(1,0);
  typ = zeros(1,0);
  cut = zeros(1,0);

  cnt = 1;

  spl(:,cnt) = r(:);
  wgt(cnt) = -1;    typ(cnt) =  9;    cut(cnt) = rhocut;    cnt=cnt+1;

  if(1)
    % local pseudopotential, as well as the derivative computed via
    % numerical differentiation
    spl(:,cnt) = Vloc(:);
    wgt(cnt) = -1;   typ(cnt)  =  99;    cut(cnt) = rhocut;    cnt=cnt+1;
    spl(:,cnt) = dVloc(:);
    wgt(cnt) = -1;   typ(cnt)  =  99;  cut(cnt) = rhocut; cnt=cnt+1;
  end

  % atomic core charge density and derivatives
  spl(:,cnt) = rhoatom(:);
  wgt(cnt) = -1;   typ(cnt)  =  999;    cut(cnt) = rhoatomcut; cnt=cnt+1;
  spl(:,cnt) = drhoatom(:);
  wgt(cnt) = -1;   typ(cnt)  =  999;  cut(cnt)   = rhoatomcut; cnt=cnt+1;



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

    spl(:,cnt) = fnval( spnl, r );
    wgt(cnt)   = pp.nonloc.dij(l,l);
    typ(cnt)   = pp.nonloc.lll(l);
    cut(cnt)   = wavcut;
    cnt = cnt+1;

    spl(:,cnt) = fnval(fnder(spnl,1),r);
    % NOT sure that this works in general
    % if( mod(pp.nonloc.lll(l),2) == 1 )
    % spl(1,cnt) = spl(2,cnt);
    % end
    wgt(cnt)   = pp.nonloc.dij(l,l);
    typ(cnt)   = pp.nonloc.lll(l);
    cut(cnt)   = wavcut;
    cnt = cnt+1;
  end


  if(1)
    fprintf('Check for normalization condition for element %3d\n', Znuc);
    fprintf('Total number of pseudopotentials terms %3d\n', cnt-1);

    % Local pseudopotential
    pk = find(r>0 & r<rhocut); 
    fprintf('int rhoGaussian = %15.10e\n', ...
      sum(4*pi*r(pk).^2.*(rhoGaussian(r(pk))).*pp.rab(pk)));

    % Model core charge density
    pk = find(r>0 & r<rhoatomcut); 
    fprintf('int rhoatom     = %15.10e\n', ...
      sum(4*pi*r(pk).^2.*(rhoatom(pk)).*pp.rab(pk)));


    % Nonlocal pseudopotential
    nlstart = 6;
    for l = 1 : pp.nonloc.nbeta
      pk = find(r>0 & r<wavcut);
      sumpp = sum(r.^2.*(spl(:,2*(l-1)+nlstart)).^2.*pp.rab);
      sumppcut = sum(r(pk).^2.*(spl(pk,2*(l-1)+nlstart)).^2.*pp.rab(pk));
      fprintf('Abs: int p^2  = %15.10e\n', sumpp);
      fprintf('Rel: int p^2  = %15.10e\n', sumppcut / sumpp );
    end

    fprintf('\n');
  end

  ess = {cur spl wgt typ cut};

  res{g,1} = Znuc;
  res{g,2} = ess;

  if(0)
    for l = 1 : size(spl,2);
      fprintf('spl(:,%d)\n',l);
      plot(r,spl(:,l),'o');xlim([0 3.0]);
      pause
    end
  end
  if(1)
    hold on
    plot(pp.r,Vlocshort,'r-.',pp.r,rhoGaussian(pp.r),'b-.');
    pause
  end
  if(0)
    plot(r,Vloc,'r-.',r,dVloc,'b-.');
    pause
  end
end

if(1)
  binstr = sprintf('ONCVVLocal.bin');
  fid = fopen(binstr, 'w');
  string = {'map', ...
    {'int'}, ...
    {'tuple', ...
    {'DblNumVec'}, ...
    {'DblNumMat'}, ...
    {'DblNumVec'}, ...
    {'IntNumVec'}, ...
    {'DblNumVec'}, ...
    }...
    };
  serialize(fid,res,string);
  fclose(fid);
end

