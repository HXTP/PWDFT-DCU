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
Znucs = [1];
res = cell(length(Znucs),2);

for g=1:length(Znucs)
  Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

  % NOTE: rhoatomcut is only chosen to achieve around 95% of the charge! 
  % For large systems it is expensive to generate the initial density in
  % the real space, and Fourier space with structure factor could be a
  % better idea.
  switch(Znuc)
    case 1 % H
      mass = 1.00794;
      rhocut  = 2.0;
      rhoatomcut = 3.5;
      wavcut  = 1.5;
%       ppFile = './ONCV_Pask/20161112/H/H.LDA.psp8';
%       pprholocFile = './ONCV_Pask/20161112/H/H.LDA.rholoc';
      ppFile = './ONCV_Pask/20170201_LDA/H/H.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/H/H.1.02.oncvpsp.rholoc';
    case 3 % Li
      mass = 6.941;
      rhocut  = 3.0;
      rhoatomcut = 6.0;
      wavcut  = 2.0;
%       ppFile = './ONCV_Pask/20161112/Li/Li.LDA.psp8';
%       pprholocFile = './ONCV_Pask/20161112/Li/Li.LDA.rholoc';
      ppFile = './ONCV_Pask/20170201_LDA/Li/Li.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/Li/Li.1.02.oncvpsp.rholoc';
    case 6 % C
      mass = 12.011;
      rhocut  = 3.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/C/C.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/C/C.1.02.oncvpsp.rholoc';
    case 8 % O
      mass = 15.9994;
      rhocut  = 3.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/O/O.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/O/O.1.02.oncvpsp.rholoc';
    case 9 % F
      mass = 18.9984032;
      rhocut  = 3.0;
      rhoatomcut = 4.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/F/F.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/F/F.1.02.oncvpsp.rholoc';
    case 13 % Al
      mass = 26.9815386;
      rhocut  = 4.0;
      rhoatomcut = 5.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/Al/Al.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/Al/Al.1.02.oncvpsp.rholoc';
    case 15 % P
      mass = 30.973762;
      rhocut  = 4.0;
      rhoatomcut = 5.0;
      wavcut  = 2.0;
      ppFile = './ONCV_Pask/20170201_LDA/P/P.1.02.oncvpsp.psp8';
      pprholocFile = './ONCV_Pask/20170201_LDA/P/P.1.02.oncvpsp.rholoc';
  end

  pp = psp8read( [], ppFile );
  if( Znuc ~= pp.anum )
    error('Znuc != pp.anum, reading the wrong pseudopotential file');
  end
  Zion = pp.venum;
  r = pp.r;

  % Interpolation grid to correctly treat the value at r=0.

  % Note rho is positive when read from THIS file
  % In the future the computation of rholoc should be derived from Vloc.
  rholocData = load(pprholocFile);
  % The following line is important for processing the data from
  % 2/1/2017
  rholocData(find(rholocData(:,1)>rhocut),2) = 0.0;
  [interpr, rhointerpr] = ...
    splinerad( rholocData(:,1), rholocData(:,2), 1 );
  sprholoc = csape(interpr, rhointerpr);

  rho = fnval(sprholoc, r);
  drho = fnval(fnder(sprholoc,1),r);
  Vloc = pp.vloc;

  % atomic core charge density. 
  % Note: rhoatom read from psp8read is multiplied by 4*pi*r^2, and drhoc
  % from psp8 is not used!
  [interpr, rhoatominterpr] = ...
    splinerad( pp.r, pp.rhoatom, 1 );
  rhoatominterpr = rhoatominterpr ./ (4*pi*interpr.^2);
  sprhoatom = csape(interpr, rhoatominterpr);
  rhoatom = fnval(sprhoatom,r);
  drhoatom = fnval(fnder(sprhoatom,1),r);

  % Note the minus sign
  Es = 1/2* sum(4*pi*r.^2 .* Vloc .* (-rho) .* pp.rab);

  cur = [Znuc mass Zion Es];

  spl = zeros(numel(r), 0);
  wgt = zeros(1,0);
  typ = zeros(1,0);
  cut = zeros(1,0);

  cnt = 1;

  spl(:,cnt) = r(:);
  wgt(cnt) = -1;    typ(cnt) =  9;    cut(cnt) = rhocut;    cnt=cnt+1;

  % local pseudopotential, as well as the derivative computed via
  % numerical differentiation
  spl(:,cnt) = rho(:);
  wgt(cnt) = -1;   typ(cnt)  =  99;    cut(cnt) = rhocut;    cnt=cnt+1;
  spl(:,cnt) = drho(:);
  wgt(cnt) = -1;   typ(cnt)  =  99;  cut(cnt) = rhocut; cnt=cnt+1;

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
    fprintf('int rho         = %15.10e\n', ...
      sum(4*pi*r(pk).^2.*(rho(pk)).*pp.rab(pk)));

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
end

if(1)
  binstr = sprintf('ONCV.bin');
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

