% Generate pseudopotential from UPF file format to the binary format
% that can be read by PWDFT/DGDFT (2016 or later).
%
Znucs = [1 8 14 47];

ppFileList = {...
  'H_ONCV_PBE-1.0.upf',...
  'Si_ONCV_PBE-1.0.upf',...
  };

numPPFile = length(ppFileList);

ppList = cell(numPPFile,1);

res = cell(numPPFile,2);

for iFile=1:numPPFile
  fprintf('Processing file %s\n', ppFileList{iFile});
  ppList{iFile} = upfread( [], ppFileList{iFile} );
  pp = ppList{iFile};
  Znuc = pp.anum;

  res{iFile,1} = Znuc;

end
% 
	% Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);
% 
	% % Initialize.  
	% % LLIN: IMPORTANT: Everthing is zero but r variables initialized to 1
	% % to avoid overflow /underflow 
	% Zion = 0;
	% mass = 0;
% 
	% % Local pseudopotential
	% rloc = 1;
	% C1   = 0;
	% C2   = 0;
	% C3   = 0;
	% C4   = 0;
% 
	% % Regular nonlocal pseudopotential
	% r0   = 1;
	% h011 = 0;
	% h022 = 0;
	% h033 = 0;
	% r1   = 1;
	% h111 = 0;
	% h122 = 0;
	% h133 = 0;
	% r2   = 1;
	% h211 = 0;
	% h222 = 0;
	% h233 = 0;
	% r3   = 1;
	% h311 = 0;
% 
	% % Spin-orbit coupling
	% k111 = 0;
	% k122 = 0;
	% k133 = 0;
	% k211 = 0;
	% k222 = 0;
	% k233 = 0; 
	% k311 = 0;
% 
% 
	% % New element template
	% % if(Znuc==)
	% % Zion = ;
	% % mass = ;
	% % rloc    = ;
	% % C1      = 
	% % k211    = ;
	% % k222    = ;
	% %
	% % rhocut  = ;
	% % wavcut  = ;
	% % end 
% 
	% % H
	% if(Znuc==1)
		% Zion = 1;
		% mass = 1.00794;
		% rloc    = 0.200000;
		% C1      = -4.178900438;
		% C2      = .724463313;
		% %
		% rhocut  = 2.0;
		% wavcut  = 2.0;
	% end 
% 
% 
	% % C
	% if(Znuc==6)
		% Zion = 4;
		% mass = 12.011;
		% rloc = .338471240;
		% C1 = -8.803673976;
		% C2 = 1.339210855;
		% r0 = .302575747;
		% h011 = 9.622486655;
		% r1 = .291506936;
		% rhocut = 3.0;
		% wavcut = 3.0;
	% end
% 
	% % O
	% if(Znuc==8)
		% Zion = 6;
		% mass = 15.9994;
		% rloc    = .244554298;
		% C1      = -16.667214800;
		% C2      = 2.487311321;
		% r0      = .220955917;
		% h011    = 18.337458110;
		% r1      = .211332471;
		% %
		% rhocut  = 2.0;
		% wavcut  = 2.0;
	% end 
% 
	% % F
	% if(Znuc==9)
		% Zion = 7;
		% mass = 18.9984032;
		% rloc    = .214929587;
		% C1      = -21.573028360;
		% C2      = 3.199776151;
		% r0      = .194684020;
		% h011    = 23.743540450;
		% r1      = .186156084;
% 
		% rhocut = 2.0;
		% wavcut = 2.0;
	% end 
% 
	% %Si
	% if(Znuc==14)
		% Zion = 4;
		% mass = 28.0855;
		% rloc = .440000000;
		% C1 = -6.269288327;
		% r0 = .435633828;
		% h011 = 8.951741502;
		% h022 = 3.493780601;
		% r1 = .497942179;
		% h111 = 2.431276734;
		% rhocut = 3.5;
		% wavcut = 3.5;
	% end
% 
% 
	% % P
	% if(Znuc==15) 
		% Zion = 5;
		% mass = 30.973762;
		% rloc = 0.430000; 
		% C1 = -5.875943266;
		% r0 = .396377422;
		% h011 = 11.008862070;
		% h022 = 4.480210422;
		% r1 = .448298381;
		% h111 = 3.056064157;
		% rhocut = 3.5;
		% wavcut = 3.5;
	% end
% 
	% % Ag
	% if(Znuc==47)
		% Zion    = 11;
		% mass    = 107.8682;
		% rloc    = 0.57000000;
		% C1      = 0.03341912;
		% r0      = 0.52704461;
		% h011    = 9.58204535;
		% h022    = 8.43071259;
		% h033    = 2.04333882;
		% r1      = 0.62911218;
		% h111    = 3.87401904;
		% h122    = 2.06129925;
		% k111    = 0.25749375;
		% k122    = 0.48677875;
		% r2      = 0.40538186;
		% h211    = -2.72495276;
		% h222    = 0.49159781;
		% k211    = -0.04679548;
		% k222    = -0.09845003;
		% rhocut = 4.5;
		% wavcut = 3.5;
  % end
% 
	% % Derived quantities
	% h012 = -1/2*sqrt(3/5)*h022;
	% h013 = 1/2 * sqrt(5/21) * h033;
	% h023 = -1/2 * sqrt(100/63) * h033;
	% h112 = -1/2 * sqrt(5/7) * h122;
	% h113 = 1/6 * sqrt(35/11) * h133;
	% h123 = -1/6 * 14 / sqrt(11) * h133;
	% h212 = -1/2 * sqrt(7/9) * h222;
	% h213 = 1/2  * sqrt(63/143) * h233;
	% h223 = -1/2 * 18 / sqrt(143) * h233;
% 
	% k112 = -1/2 * sqrt(5/7) * k122;
	% k113 = 1/6 * sqrt(35/11) * k133;
	% k123 = -1/6 * 14 / sqrt(11) * k133;
	% k212 = -1/2 * sqrt(7/9) * k222;
	% k213 = 1/2  * sqrt(63/143) * k233;
	% k223 = -1/2 * 18 / sqrt(143) * k233;
% 
% 
	% % Functional form for pseudopotentials
	% Vlocfn = @(r) ...
		% (-Zion./r.*erf(r./(sqrt(2)*rloc)) + ...
		% exp(-1/2*(r./rloc).^2).*(C1+C2*(r/rloc).^2+C3*(r/ ...
		% rloc).^4+C4*(r./rloc).^6));
% 
	% rho0fn = @(r) ...
		% (1/(4*pi).*(-sqrt(2/pi)*Zion./rloc.^3.*exp(-1/2*(r./rloc).^2) ...
		% + exp(-1/2*(r./rloc).^2).* ...
		% ((-6*C2+3*C1)/rloc.^2 + (-20*C3+7*C2-C1).*r.^2/rloc.^4 ...
		% + (-42*C4+11*C3-C2).*r.^4/rloc.^6 + ...
		% (15*C4-C3).*r.^6/rloc.^8 - C4.*r.^8./rloc.^10)));
% 
	% drho0fn = @(r) ...
		% (1/(4*pi).*(sqrt(2/pi)*Zion*r./rloc.^5.*exp(-1/2*(r./rloc).^2) ...
		% + exp(-1/2*(r./rloc).^2).* ...
		% ((-40*C3+20*C2-5*C1)*r/rloc.^4 + ...
		% (-168*C4+64*C3-11*C2+C1).*r.^3/rloc.^6 + ...
		% (132*C4-17*C3+C2).*r.^5/rloc.^8 + ...
		% (-23*C4+C3).*r.^7/rloc.^10 + C4.*r.^9./ ...
		% rloc.^12)));
% 
% 
	% l=0; i=1; rl=r0;
	% p01fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=0; i=2; rl=r0;
	% p02fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=0; i=3; rl=r0;
	% p03fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=1; i=1; rl=r1;
	% p11fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=1; i=2; rl=r1;
	% p12fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=1; i=3; rl=r1;
	% p13fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=2; i=1; rl=r2;
	% p21fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=2; i=2; rl=r2;
	% p22fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=2; i=3; rl=r2;
	% p23fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
	% l=3; i=1; rl=r3;
	% p31fn = @(r) ...
		% (sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		% (rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));
% 
% 
	% % Diagonalize the h coefficient matrices to obtain regular pseudopotential
	% % l=0 channel
	% tmp = [h011 h012 h013;
	% h012 h022 h023;
	% h013 h023 h033];
	% [V0,D0] = eig(tmp);
	% D0 = diag(D0);  [D0, idx] = sort(D0,'descend'); V0 = V0(:,idx);
% 
	% p0afn = @(r) (p01fn(r)*V0(1,1) + p02fn(r)*V0(2,1) + p03fn(r)*V0(3,1));
	% h0a = D0(1);
% 
	% p0bfn = @(r) (p01fn(r)*V0(1,2) + p02fn(r)*V0(2,2) + p03fn(r)*V0(3,2));
	% h0b = D0(2);
% 
	% p0cfn = @(r) (p01fn(r)*V0(1,3) + p02fn(r)*V0(2,3) + p03fn(r)*V0(3,3));
	% h0c = D0(3);
% 
	% % l=1 channel
	% tmp = [h111 h112 h113;
	% h112 h122 h123;
	% h113 h123 h133];
	% [V1,D1] = eig(tmp);
	% D1 = diag(D1);  [D1, idx] = sort(D1,'descend'); V1 = V1(:,idx);
% 
	% p1afn = @(r) (p11fn(r)*V1(1,1) + p12fn(r)*V1(2,1) + p13fn(r)*V1(3,1));
	% h1a   = D1(1);
% 
	% p1bfn = @(r) (p11fn(r)*V1(1,2) + p12fn(r)*V1(2,2) + p13fn(r)*V1(3,2));
	% h1b   = D1(2); 
% 
	% p1cfn = @(r) (p11fn(r)*V1(1,3) + p12fn(r)*V1(2,3) + p13fn(r)*V1(3,3));
	% h1c   = D1(3);
% 
	% % l=2 channel. 
	% tmp = [h211 h212 h213;
	% h212 h222 h223;
	% h213 h223 h233];
% 
	% [V2,D2] = eig(tmp);
	% D2 = diag(D2);  [D2, idx] = sort(D2,'descend'); V2 = V2(:,idx);
% 
	% p2afn = @(r) (p21fn(r)*V2(1,1) + p22fn(r)*V2(2,1) + p23fn(r)*V2(3,1));
	% h2a   = D2(1);
% 
	% p2bfn = @(r) (p21fn(r)*V2(1,2) + p22fn(r)*V2(2,2) + p23fn(r)*V2(3,2));
	% h2b   = D2(2);
% 
	% p2cfn = @(r) (p21fn(r)*V2(1,3) + p22fn(r)*V2(2,3) + p23fn(r)*V2(3,3));
	% h2c   = D2(3); 
% 
	% % l=3 is special
	% p3afn = @(r) p31fn(r);
	% h3a   = h311; 
% 
% 
	% % Diagonalize the k coefficient matrices to obtain pseudopotential for
	% % spin-orbit coupling
% 
	% % l=1 channel
	% tmp = [k111 k112 k113;
	% k112 k122 k123;
	% k113 k123 k133];
	% [Vk1,Dk1] = eig(tmp);
	% Dk1 = diag(Dk1);  [Dk1, idx] = sort(Dk1,'descend'); Vk1 = Vk1(:,idx);
% 
	% p1kafn = @(r) (p11fn(r)*Vk1(1,1) + p12fn(r)*Vk1(2,1) + p13fn(r)*Vk1(3,1));
	% k1a   = Dk1(1);
% 
	% p1kbfn = @(r) (p11fn(r)*Vk1(1,2) + p12fn(r)*Vk1(2,2) + p13fn(r)*Vk1(3,2));
	% k1b   = Dk1(2); 
% 
	% p1kcfn = @(r) (p11fn(r)*Vk1(1,3) + p12fn(r)*Vk1(2,3) + p13fn(r)*Vk1(3,3));
	% k1c   = Dk1(3);
% 
	% % l=2 channel. 
	% tmp = [k211 k212 k213;
	% k212 k222 k223;
	% k213 k223 k233];
% 
	% [Vk2,Dk2] = eig(tmp);
	% Dk2 = diag(Dk2);  [Dk2, idx] = sort(Dk2,'descend'); Vk2 = Vk2(:,idx);
% 
	% p2kafn = @(r) (p21fn(r)*Vk2(1,1) + p22fn(r)*Vk2(2,1) + p23fn(r)*Vk2(3,1));
	% k2a   = Dk2(1);
% 
	% p2kbfn = @(r) (p21fn(r)*Vk2(1,2) + p22fn(r)*Vk2(2,2) + p23fn(r)*Vk2(3,2));
	% k2b   = Dk2(2);
% 
	% p2kcfn = @(r) (p21fn(r)*Vk2(1,3) + p22fn(r)*Vk2(2,3) + p23fn(r)*Vk2(3,3));
	% k2c    = Dk2(3); 
% 
	% % l=3 is special
	% p3kafn = @(r) p31fn(r);
	% k3a    = k311; 
% 
% 
	% %sample
	% stp = 0.002;
	% r = [-10+stp/2:stp:10];
% 
	% Vloc = Vlocfn(r);
	% p0a = p0afn(r);
	% p0b = p0bfn(r);
	% p0c = p0cfn(r);
	% p1a = p1afn(r);
	% p1b = p1bfn(r);
	% p1c = p1cfn(r);
	% p2a = p2afn(r);
	% p2b = p2bfn(r);
	% p2c = p2cfn(r);
	% p3a = p3afn(r);
% 
	% p1ka = p1kafn(r);
	% p1kb = p1kbfn(r);
	% p1kc = p1kcfn(r);
	% p2ka = p2kafn(r);
	% p2kb = p2kbfn(r);
	% p2kc = p2kcfn(r);
	% p3ka = p3kafn(r);
% 
	% p0a_pp = csape(r,p0a);
	% p0b_pp = csape(r,p0b);
	% p0c_pp = csape(r,p0c);
	% p1a_pp = csape(r,p1a);
	% p1b_pp = csape(r,p1b);
	% p1c_pp = csape(r,p1c);
	% p2a_pp = csape(r,p2a);
	% p2b_pp = csape(r,p2b);
	% p2c_pp = csape(r,p2c);
	% p3a_pp = csape(r,p3a);
% 
	% p1ka_pp = csape(r,p1ka);
	% p1kb_pp = csape(r,p1kb);
	% p1kc_pp = csape(r,p1kc);
	% p2ka_pp = csape(r,p2ka);
	% p2kb_pp = csape(r,p2kb);
	% p2kc_pp = csape(r,p2kc);
	% p3ka_pp = csape(r,p3ka);
% 
% 
	% % Pseudo-charge and its derivatives
	% rho = rho0fn(r); 
	% drho = drho0fn(r); 
% 
	% % Regular nonlocal pseudopotential and their derivatives
	% p0a_dpp = fnder(p0a_pp,1);
	% dp0a = fnval(p0a_dpp,r);
% 
	% p0b_dpp = fnder(p0b_pp,1);
	% dp0b = fnval(p0b_dpp,r);
% 
	% p0c_dpp = fnder(p0c_pp,1);
	% dp0c = fnval(p0c_dpp,r);
% 
	% p1a_dpp = fnder(p1a_pp,1);
	% dp1a = fnval(p1a_dpp,r);
% 
	% p1b_dpp = fnder(p1b_pp,1);
	% dp1b = fnval(p1b_dpp,r);
% 
	% p1c_dpp = fnder(p1c_pp,1);
	% dp1c = fnval(p1c_dpp,r);
% 
	% p2a_dpp = fnder(p2a_pp,1);
	% dp2a = fnval(p2a_dpp,r);
% 
	% p2b_dpp = fnder(p2b_pp,1);
	% dp2b = fnval(p2b_dpp,r);
% 
	% p2c_dpp = fnder(p2c_pp,1);
	% dp2c = fnval(p2c_dpp,r);
% 
	% p3a_dpp = fnder(p3a_pp,1);
	% dp3a = fnval(p3a_dpp,r);
% 
	% % Nonlocal pseudopotential for spin-orbit coupling and their
	% % derivatives
	% p1ka_dpp = fnder(p1ka_pp,1);
	% dp1ka = fnval(p1ka_dpp,r);
% 
	% p1kb_dpp = fnder(p1kb_pp,1);
	% dp1kb = fnval(p1kb_dpp,r);
% 
	% p1kc_dpp = fnder(p1kc_pp,1);
	% dp1kc = fnval(p1kc_dpp,r);
% 
	% p2ka_dpp = fnder(p2ka_pp,1);
	% dp2ka = fnval(p2ka_dpp,r);
% 
	% p2kb_dpp = fnder(p2kb_pp,1);
	% dp2kb = fnval(p2kb_dpp,r);
% 
	% p2kc_dpp = fnder(p2kc_pp,1);
	% dp2kc = fnval(p2kc_dpp,r);
% 
	% p3ka_dpp = fnder(p3ka_pp,1);
	% dp3ka = fnval(p3ka_dpp,r);
% 
	% gd = find(r>0);
	% Es = 1/2* sum(4*pi*r(gd).^2 .* Vloc(gd) .* rho(gd) * stp);
% 
	% cur = [Znuc mass Zion Es];
% 
	% spl = zeros(numel(r), 0);
	% wgt = zeros(1,0);
	% typ = zeros(1,0);
	% cut = zeros(1,0);
% 
	% cnt = 1;
% 
% 
	% spl(:,cnt) = r(:);
	% wgt(cnt) = -1;    typ(cnt) =  9;    cut(cnt) = rhocut;    cnt=cnt+1;
% 
	% spl(:,cnt) = -rho(:);
	% wgt(cnt) = -1;   typ(cnt)  =  99;    cut(cnt) = rhocut;    cnt=cnt+1;
	% spl(:,cnt) = -drho(:);
	% wgt(cnt) = -1;   typ(cnt)  =  99;  cut(cnt) = rhocut; cnt=cnt+1;
% 
	% if(abs(h0a) > 1e-10)
		% spl(:,cnt) = p0a(:);
		% wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp0a(:);
		% wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h0b) > 1e-10)
		% spl(:,cnt) = p0b(:);
		% wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp0b(:);
		% wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h0c) > 1e-10)
		% spl(:,cnt) = p0c(:);
		% wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp0c(:);
		% wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h1a) > 1e-10)
		% spl(:,cnt) = p1a(:);
		% wgt(cnt) = h1a;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1a(:);
		% wgt(cnt) = h1a;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h1b) > 1e-10)
		% spl(:,cnt) = p1b(:);
		% wgt(cnt) = h1b;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1b(:);
		% wgt(cnt) = h1b;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h1c) > 1e-10)
		% spl(:,cnt) = p1c(:);
		% wgt(cnt) = h1c;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1c(:);
		% wgt(cnt) = h1c;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h2a) > 1e-10)
		% spl(:,cnt) = p2a(:);
		% wgt(cnt) = h2a;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2a(:);
		% wgt(cnt) = h2a;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h2b) > 1e-10)
		% spl(:,cnt) = p2b(:);
		% wgt(cnt) = h2b;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2b(:);
		% wgt(cnt) = h2b;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h2c) > 1e-10)
		% spl(:,cnt) = p2c(:);
		% wgt(cnt) = h2c;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2c(:);
		% wgt(cnt) = h2c;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(h3a) > 1e-10)
		% spl(:,cnt) = p3a(:);
		% wgt(cnt) = h3a;   typ(cnt) = 3;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp3a(:);
		% wgt(cnt) = h3a;   typ(cnt) = 3;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k1a) > 1e-10)
		% spl(:,cnt) = p1ka(:);
		% wgt(cnt) = k1a;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1ka(:);
		% wgt(cnt) = k1a;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k1b) > 1e-10)
		% spl(:,cnt) = p1kb(:);
		% wgt(cnt) = k1b;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1kb(:);
		% wgt(cnt) = k1b;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k1c) > 1e-10)
		% spl(:,cnt) = p1kc(:);
		% wgt(cnt) = k1c;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp1kc(:);
		% wgt(cnt) = k1c;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k2a) > 1e-10)
		% spl(:,cnt) = p2ka(:);
		% wgt(cnt) = k2a;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2ka(:);
		% wgt(cnt) = k2a;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k2b) > 1e-10)
		% spl(:,cnt) = p2kb(:);
		% wgt(cnt) = k2b;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2kb(:);
		% wgt(cnt) = k2b;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k2c) > 1e-10)
		% spl(:,cnt) = p2kc(:);
		% wgt(cnt) = k2c;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp2kc(:);
		% wgt(cnt) = k2c;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
	% if(abs(k3a) > 1e-10)
		% spl(:,cnt) = p3ka(:);
		% wgt(cnt) = k3a;   typ(cnt) = -3;  cut(cnt) = wavcut; cnt=cnt+1;
		% spl(:,cnt) = dp3ka(:);
		% wgt(cnt) = k3a;   typ(cnt) = -3;  cut(cnt) = wavcut; cnt=cnt+1;
	% end
% 
% 
	% if(1)
		% fprintf('Check for normalization condition for element %3d\n', Znuc);
		% fprintf('Total number of pseudopotentials terms %3d\n', cnt-1);
% 
		% % Local pseudopotential
		% pk = find(r>0 & r<rhocut); 
		% fprintf('int rho         = %15.10e\n', sum(4*pi*r(pk).^2.*(-rho(pk))*stp) );
% 
		% % Nonlocal pseudopotential
		% pk = find(r>0 & r<wavcut);
		% if( abs(h0a) > 1e-10 )
			% fprintf('Abs: int p0a^2  = %15.10e\n', sum(r(pk).^2.*p0a(pk).^2*stp) );
			% fprintf('Rel: int p0a^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p0a(pk).^2*stp) / (sum(r.^2.*p0a.^2*stp)/2) );
		% end
		% if( abs(h0b) > 1e-10 )
			% fprintf('Abs: int p0b^2  = %15.10e\n', sum(r(pk).^2.*p0b(pk).^2*stp) );
			% fprintf('Rel: int p0b^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p0b(pk).^2*stp) / (sum(r.^2.*p0b.^2*stp)/2) );
		% end
		% if( abs(h0c) > 1e-10 )
			% fprintf('Abs: int p0c^2  = %15.10e\n', sum(r(pk).^2.*p0c(pk).^2*stp) );
			% fprintf('Rel: int p0c^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p0c(pk).^2*stp) / (sum(r.^2.*p0c.^2*stp)/2) );
		% end
		% if( abs(h1a) > 1e-10 )
			% fprintf('Abs: int p1a^2  = %15.10e\n', sum(r(pk).^2.*p1a(pk).^2*stp) );
			% fprintf('Rel: int p1a^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1a(pk).^2*stp) / (sum(r.^2.*p1a.^2*stp)/2) );
		% end
		% if( abs(h1b) > 1e-10 )
			% fprintf('Abs: int p1b^2  = %15.10e\n', sum(r(pk).^2.*p1b(pk).^2*stp) );
			% fprintf('Rel: int p1b^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1b(pk).^2*stp) / (sum(r.^2.*p1b.^2*stp)/2) );
		% end
		% if( abs(h1c) > 1e-10 )
			% fprintf('Abs: int p1c^2  = %15.10e\n', sum(r(pk).^2.*p1c(pk).^2*stp) );
			% fprintf('Rel: int p1c^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1c(pk).^2*stp) / (sum(r.^2.*p1c.^2*stp)/2) );
		% end
		% if( abs(h2a) > 1e-10 )
			% fprintf('Abs: int p2a^2  = %15.10e\n', sum(r(pk).^2.*p2a(pk).^2*stp) );
			% fprintf('Rel: int p2a^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2a(pk).^2*stp) / (sum(r.^2.*p2a.^2*stp)/2) );
		% end
		% if( abs(h2b) > 1e-10 )
			% fprintf('Abs: int p2b^2  = %15.10e\n', sum(r(pk).^2.*p2b(pk).^2*stp) );
			% fprintf('Rel: int p2b^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2b(pk).^2*stp) / (sum(r.^2.*p2b.^2*stp)/2) );
		% end
		% if( abs(h2c) > 1e-10 )
			% fprintf('Abs: int p2c^2  = %15.10e\n', sum(r(pk).^2.*p2c(pk).^2*stp) );
			% fprintf('Rel: int p2c^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2c(pk).^2*stp) / (sum(r.^2.*p2c.^2*stp)/2) );
		% end
		% if( abs(h3a) > 1e-10 )
			% fprintf('Abs: int p3a^2  = %15.10e\n', sum(r(pk).^2.*p3a(pk).^2*stp) );
			% fprintf('Rel: int p3a^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p3a(pk).^2*stp) / (sum(r.^2.*p3a.^2*stp)/2) );
		% end
% 
		% if( abs(k1a) > 1e-10 )
			% fprintf('Abs: int p1ka^2  = %15.10e\n', sum(r(pk).^2.*p1ka(pk).^2*stp) );
			% fprintf('Rel: int p1ka^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1ka(pk).^2*stp) / (sum(r.^2.*p1ka.^2*stp)/2) );
		% end
% 
		% if( abs(k1b) > 1e-10 )
			% fprintf('Abs: int p1kb^2  = %15.10e\n', sum(r(pk).^2.*p1kb(pk).^2*stp) );
			% fprintf('Rel: int p1kb^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1kb(pk).^2*stp) / (sum(r.^2.*p1kb.^2*stp)/2) );
		% end
% 
		% if( abs(k1c) > 1e-10 )
			% fprintf('Abs: int p1kc^2  = %15.10e\n', sum(r(pk).^2.*p1kc(pk).^2*stp) );
			% fprintf('Rel: int p1kc^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p1kc(pk).^2*stp) / (sum(r.^2.*p1kc.^2*stp)/2) );
		% end
% 
		% if( abs(k2a) > 1e-10 )
			% fprintf('Abs: int p2ka^2  = %15.10e\n', sum(r(pk).^2.*p2ka(pk).^2*stp) );
			% fprintf('Rel: int p2ka^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2ka(pk).^2*stp) / (sum(r.^2.*p2ka.^2*stp)/2) );
		% end
% 
		% if( abs(k2b) > 1e-10 )
			% fprintf('Abs: int p2kb^2  = %15.10e\n', sum(r(pk).^2.*p2kb(pk).^2*stp) );
			% fprintf('Rel: int p2kb^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2kb(pk).^2*stp) / (sum(r.^2.*p2kb.^2*stp)/2) );
		% end
% 
		% if( abs(k2c) > 1e-10 )
			% fprintf('Abs: int p2kc^2  = %15.10e\n', sum(r(pk).^2.*p2kc(pk).^2*stp) );
			% fprintf('Rel: int p2kc^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p2kc(pk).^2*stp) / (sum(r.^2.*p2kc.^2*stp)/2) );
		% end
% 
		% if( abs(k3a) > 1e-10 )
			% fprintf('Abs: int p3ka^2  = %15.10e\n', sum(r(pk).^2.*p3ka(pk).^2*stp) );
			% fprintf('Rel: int p3ka^2  = %15.10e\n', ...
				% sum(r(pk).^2.*p3ka(pk).^2*stp) / (sum(r.^2.*p3ka.^2*stp)/2) );
		% end
% 
		% fprintf('\n');
	% end
% 
	% ess = {cur spl wgt typ cut};
% 
	% res{g,1} = Znuc;
	% res{g,2} = ess;
% 
% end
% 
% if(1)
	% binstr = sprintf('HGH.bin');
	% fid = fopen(binstr, 'w');
	% string = {'map', ...
		% {'int'}, ...
		% {'tuple', ...
		% {'DblNumVec'}, ...
		% {'DblNumMat'}, ...
		% {'DblNumVec'}, ...
		% {'IntNumVec'}, ...
		% {'DblNumVec'}, ...
		% }...
		% };
	% serialize(fid,res,string);
	% fclose(fid);
% end
% 
% if(1)
	% fid = fopen(binstr,'r');
	% restst = deserialize(fid, string);
	% fclose(fid);
% end
