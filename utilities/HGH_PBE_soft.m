% Generation of the soft HGH pseudopotential WITHOUT the nonlocal core
% correction.
%
% Obtained from Santanu Saha, 2014/2/16
% 
% LLIN: 
%
Znucs = [1 3 6 7 8 9 14 15 22];
% Znucs = [22];

res = cell(length(Znucs),2);

for g=1:length(Znucs)

	Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

	% Initialize.  
	% LLIN: IMPORTANT: Everthing is zero but r variables initialized to 1
	% to avoid overflow /underflow 
	Zion = 0;
	mass = 0;

	% Local pseudopotential
	rloc = 1;
	C1   = 0;
	C2   = 0;
	C3   = 0;
	C4   = 0;

	% Regular nonlocal pseudopotential
	r0   = 1;
	h011 = 0;
  h012 = 0;
  h013 = 0;
	h022 = 0;
  h023 = 0;
	h033 = 0;
	r1   = 1;
	h111 = 0;
  h112 = 0;
  h113 = 0;
	h122 = 0;
  h123 = 0;
	h133 = 0;
	r2   = 1;
	h211 = 0;
  h212 = 0;
  h213 = 0;
	h222 = 0;
  h223 = 0;
	h233 = 0;
	r3   = 1;
	h311 = 0;

	% Spin-orbit coupling
	k111 = 0;
  k112 = 0;
  k113 = 0;
	k122 = 0;
  k123 = 0;
	k133 = 0;
	k211 = 0;
  k212 = 0;
  k213 = 0;
	k222 = 0;
  k223 = 0;
	k233 = 0; 
	k311 = 0;


	% New element template
	% if(Znuc==)
	% Zion = ;
	% mass = ;
	% rloc    = ;
	% C1      = ;
	% r0      = ;
	% h011    = ;
	% h022    = ;
	% r1      = ;
	% h111    = ;
	% h122    = ;
	% k111    = ;
	% k122    = ;
	% k133    = ;
	% r2      = ;
	% h211    = ;
	% h222    = ;
	% k211    = ;
	% k222    = ;
	%
	% rhocut  = ;
	% wavcut  = ;
	% end 

	% H 9/7/2015
	if(Znuc==1)
		Zion = 1;
		mass = 1.00794;
		rloc    = 0.228727444559;
		C1      = -3.722900963668;
		C2      = 0.657596429179;
		%
		rhocut  = 2.0;
		wavcut  = 2.0;
	end 

	% Li. Semicore. 9/9/2015
	if(Znuc==3)
		Zion = 3;
		mass = 6.941;
		rloc    = 0.4800;
		C1      = -12.633331;
		C2      = 9.4431928;
    C3      = -1.8872548;
    C4      = 0.096453468;
		%
		rhocut  = 4.5;
		wavcut  = 3.5;
	end 


  % C. 9/9/2015
	if(Znuc==6)
		Zion = 4;
		mass = 12.011;
		rloc = 0.41329225352;
		C1 = -5.729305064;
		C2 = 0.874750285642;
		r0 = 0.4341053017;
		h011 = 9.01955484658;
    h012 = -2.4185793675;
    h022 = 0.554729349186;
		rhocut = 4.0;
		wavcut = 3.0;
	end

                       
	% N. 9/9/2015
	if(Znuc==7)
		Zion = 5;
		mass = 14.007;
		rloc = 0.40469225352;
		C1 = -8.10098063736025;
		C2 = 1.294191224;
		r0 = 0.4256053017;
		h011 = 9.2183335485526;
    h012 = -2.4905918767;
    h022 = 0.38356320566013;
		rhocut = 4.5;
		wavcut = 2.5;
	end

	% O. 9/7/2015
	if(Znuc==8)
		Zion = 6;
		mass = 15.9994;
		rloc    = 0.3454999999;
		C1      = -11.7435870154;
		C2      = 1.90653967947;
		r0      = 0.368040785471;
		h011    = 10.85890680355;
    h012    = -2.1290113946;
    h022    = -0.429767298715;

		%
		rhocut  = 4.0;
		wavcut  = 3.0;
	end 
                      

	% F. 9/9/2015
	if(Znuc==9)
		Zion = 7;
		mass = 18.9984032;
		rloc    = 0.30270764726;
		C1      = -15.360747916;
		C2      = 2.453840389;
		r0      = 0.318976059;
		h011    = 14.7072505678;
    h012    = -2.9854852662;
    h022    = -0.412166422511;

		rhocut = 4.0;
		wavcut = 3.0;
	end 



	%Si. 9/9/2015
	if(Znuc==14)
		Zion = 4;
		mass = 28.0855;
		rloc = 0.55318460771;
		C1 = -3.88383097485;
    C2 = 0.3904838172143;
		r0 = 0.41164534845;
		h011 = 5.4113418416;
		h012 = -1.6000061332;
    h022 = 4.01974003784;
		r1 = 0.49248082589;
		h111 = 2.4379425883056;
    h112 = 0.00000000000000000E+00;
    h122 = 0.00000000000000000E+00;

		rhocut = 4.5;
		wavcut = 3.5;
	end


	% P. 9/9/2015
	if(Znuc==15) 
		Zion = 5;
		mass = 30.973762;
		rloc = 0.47302706024; 
		C1 = -5.64603301035;
    C2 = 0.6442592294;
		r0 = 0.373360419343;
		h011 = 4.62035081029;
    h012 = -0.63842593734;
		h022 = 4.2189516133042;
		r1 = 0.4278715404;
		h111 = 3.53692299847105;
    h112 = 0.00000000000000000E+00;
    h122 = 0.00000000000000000E+00;
		rhocut = 4.5;
		wavcut = 3.5;
	end

	% Ti (WITH semicore). 
	if(Znuc==22) 
		Zion = 12;
		mass = 47.867;
		rloc = 0.47530203999999998E+00; 
		C1 = 0.25823179904587629E+02;
    C2 = -0.71647281039614930E+01;
    C3 = 0.68072917918672193E+00;
		r0 = 0.38713492568354607E+00;
		h011 = -0.11807493902497923E+02;
		h012 = 0.49334024807558414E+01;
    h022 = -0.45893052536354579E+01;
		r1 = 0.37018003147648104E+00;
		h111 = 0.41721624590602531E+01;
    h112 = -0.10182909865833640E+02;
    h122 = 0.94773568870172475E+01;
		r2      = 0.32578815000454181E+00;
		h211    = -0.95085327236822064E+01;
		rhocut = 4.5;
		wavcut = 3.5;
	end


	% Derived quantities


	% Functional form for pseudopotentials
	Vlocfn = @(r) ...
		(-Zion./r.*erf(r./(sqrt(2)*rloc)) + ...
		exp(-1/2*(r./rloc).^2).*(C1+C2*(r/rloc).^2+C3*(r/ ...
		rloc).^4+C4*(r./rloc).^6));

	rho0fn = @(r) ...
		(1/(4*pi).*(-sqrt(2/pi)*Zion./rloc.^3.*exp(-1/2*(r./rloc).^2) ...
		+ exp(-1/2*(r./rloc).^2).* ...
		((-6*C2+3*C1)/rloc.^2 + (-20*C3+7*C2-C1).*r.^2/rloc.^4 ...
		+ (-42*C4+11*C3-C2).*r.^4/rloc.^6 + ...
		(15*C4-C3).*r.^6/rloc.^8 - C4.*r.^8./rloc.^10)));

	drho0fn = @(r) ...
		(1/(4*pi).*(sqrt(2/pi)*Zion*r./rloc.^5.*exp(-1/2*(r./rloc).^2) ...
		+ exp(-1/2*(r./rloc).^2).* ...
		((-40*C3+20*C2-5*C1)*r/rloc.^4 + ...
		(-168*C4+64*C3-11*C2+C1).*r.^3/rloc.^6 + ...
		(132*C4-17*C3+C2).*r.^5/rloc.^8 + ...
		(-23*C4+C3).*r.^7/rloc.^10 + C4.*r.^9./ ...
		rloc.^12)));


	l=0; i=1; rl=r0;
	p01fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=0; i=2; rl=r0;
	p02fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=0; i=3; rl=r0;
	p03fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=1; i=1; rl=r1;
	p11fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=1; i=2; rl=r1;
	p12fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=1; i=3; rl=r1;
	p13fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=2; i=1; rl=r2;
	p21fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=2; i=2; rl=r2;
	p22fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=2; i=3; rl=r2;
	p23fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));

	l=3; i=1; rl=r3;
	p31fn = @(r) ...
		(sqrt(2) * r.^(l+2*(i-1)).*exp(-r.^2/(2*rl^2))) ./...
		(rl.^(l+(4*i-1)/2).*sqrt(gamma(l+(4*i-1)/2)));


	% Diagonalize the h coefficient matrices to obtain regular pseudopotential
	% l=0 channel
	tmp = [h011 h012 h013;
	h012 h022 h023;
	h013 h023 h033];
	[V0,D0] = eig(tmp);
	D0 = diag(D0);  [D0, idx] = sort(D0,'descend'); V0 = V0(:,idx);

	p0afn = @(r) (p01fn(r)*V0(1,1) + p02fn(r)*V0(2,1) + p03fn(r)*V0(3,1));
	h0a = D0(1);

	p0bfn = @(r) (p01fn(r)*V0(1,2) + p02fn(r)*V0(2,2) + p03fn(r)*V0(3,2));
	h0b = D0(2);

	p0cfn = @(r) (p01fn(r)*V0(1,3) + p02fn(r)*V0(2,3) + p03fn(r)*V0(3,3));
	h0c = D0(3);

	% l=1 channel
	tmp = [h111 h112 h113;
	h112 h122 h123;
	h113 h123 h133];
	[V1,D1] = eig(tmp);
	D1 = diag(D1);  [D1, idx] = sort(D1,'descend'); V1 = V1(:,idx);

	p1afn = @(r) (p11fn(r)*V1(1,1) + p12fn(r)*V1(2,1) + p13fn(r)*V1(3,1));
	h1a   = D1(1);

	p1bfn = @(r) (p11fn(r)*V1(1,2) + p12fn(r)*V1(2,2) + p13fn(r)*V1(3,2));
	h1b   = D1(2); 

	p1cfn = @(r) (p11fn(r)*V1(1,3) + p12fn(r)*V1(2,3) + p13fn(r)*V1(3,3));
	h1c   = D1(3);

	% l=2 channel. 
	tmp = [h211 h212 h213;
	h212 h222 h223;
	h213 h223 h233];

	[V2,D2] = eig(tmp);
	D2 = diag(D2);  [D2, idx] = sort(D2,'descend'); V2 = V2(:,idx);

	p2afn = @(r) (p21fn(r)*V2(1,1) + p22fn(r)*V2(2,1) + p23fn(r)*V2(3,1));
	h2a   = D2(1);

	p2bfn = @(r) (p21fn(r)*V2(1,2) + p22fn(r)*V2(2,2) + p23fn(r)*V2(3,2));
	h2b   = D2(2);

	p2cfn = @(r) (p21fn(r)*V2(1,3) + p22fn(r)*V2(2,3) + p23fn(r)*V2(3,3));
	h2c   = D2(3); 

	% l=3 is special
	p3afn = @(r) p31fn(r);
	h3a   = h311; 


	% Diagonalize the k coefficient matrices to obtain pseudopotential for
	% spin-orbit coupling

	% l=1 channel
	tmp = [k111 k112 k113;
	k112 k122 k123;
	k113 k123 k133];
	[Vk1,Dk1] = eig(tmp);
	Dk1 = diag(Dk1);  [Dk1, idx] = sort(Dk1,'descend'); Vk1 = Vk1(:,idx);

	p1kafn = @(r) (p11fn(r)*Vk1(1,1) + p12fn(r)*Vk1(2,1) + p13fn(r)*Vk1(3,1));
	k1a   = Dk1(1);

	p1kbfn = @(r) (p11fn(r)*Vk1(1,2) + p12fn(r)*Vk1(2,2) + p13fn(r)*Vk1(3,2));
	k1b   = Dk1(2); 

	p1kcfn = @(r) (p11fn(r)*Vk1(1,3) + p12fn(r)*Vk1(2,3) + p13fn(r)*Vk1(3,3));
	k1c   = Dk1(3);

	% l=2 channel. 
	tmp = [k211 k212 k213;
	k212 k222 k223;
	k213 k223 k233];

	[Vk2,Dk2] = eig(tmp);
	Dk2 = diag(Dk2);  [Dk2, idx] = sort(Dk2,'descend'); Vk2 = Vk2(:,idx);

	p2kafn = @(r) (p21fn(r)*Vk2(1,1) + p22fn(r)*Vk2(2,1) + p23fn(r)*Vk2(3,1));
	k2a   = Dk2(1);

	p2kbfn = @(r) (p21fn(r)*Vk2(1,2) + p22fn(r)*Vk2(2,2) + p23fn(r)*Vk2(3,2));
	k2b   = Dk2(2);

	p2kcfn = @(r) (p21fn(r)*Vk2(1,3) + p22fn(r)*Vk2(2,3) + p23fn(r)*Vk2(3,3));
	k2c    = Dk2(3); 

	% l=3 is special
	p3kafn = @(r) p31fn(r);
	k3a    = k311; 


	%sample
	stp = 0.002;
	r = [-10+stp/2:stp:10];

	Vloc = Vlocfn(r);
	p0a = p0afn(r);
	p0b = p0bfn(r);
	p0c = p0cfn(r);
	p1a = p1afn(r);
	p1b = p1bfn(r);
	p1c = p1cfn(r);
	p2a = p2afn(r);
	p2b = p2bfn(r);
	p2c = p2cfn(r);
	p3a = p3afn(r);

	p1ka = p1kafn(r);
	p1kb = p1kbfn(r);
	p1kc = p1kcfn(r);
	p2ka = p2kafn(r);
	p2kb = p2kbfn(r);
	p2kc = p2kcfn(r);
	p3ka = p3kafn(r);

	p0a_pp = csape(r,p0a);
	p0b_pp = csape(r,p0b);
	p0c_pp = csape(r,p0c);
	p1a_pp = csape(r,p1a);
	p1b_pp = csape(r,p1b);
	p1c_pp = csape(r,p1c);
	p2a_pp = csape(r,p2a);
	p2b_pp = csape(r,p2b);
	p2c_pp = csape(r,p2c);
	p3a_pp = csape(r,p3a);

	p1ka_pp = csape(r,p1ka);
	p1kb_pp = csape(r,p1kb);
	p1kc_pp = csape(r,p1kc);
	p2ka_pp = csape(r,p2ka);
	p2kb_pp = csape(r,p2kb);
	p2kc_pp = csape(r,p2kc);
	p3ka_pp = csape(r,p3ka);


	% Pseudo-charge and its derivatives
	rho = rho0fn(r); 
	drho = drho0fn(r); 

	% Regular nonlocal pseudopotential and their derivatives
	p0a_dpp = fnder(p0a_pp,1);
	dp0a = fnval(p0a_dpp,r);

	p0b_dpp = fnder(p0b_pp,1);
	dp0b = fnval(p0b_dpp,r);

	p0c_dpp = fnder(p0c_pp,1);
	dp0c = fnval(p0c_dpp,r);

	p1a_dpp = fnder(p1a_pp,1);
	dp1a = fnval(p1a_dpp,r);

	p1b_dpp = fnder(p1b_pp,1);
	dp1b = fnval(p1b_dpp,r);

	p1c_dpp = fnder(p1c_pp,1);
	dp1c = fnval(p1c_dpp,r);

	p2a_dpp = fnder(p2a_pp,1);
	dp2a = fnval(p2a_dpp,r);

	p2b_dpp = fnder(p2b_pp,1);
	dp2b = fnval(p2b_dpp,r);

	p2c_dpp = fnder(p2c_pp,1);
	dp2c = fnval(p2c_dpp,r);

	p3a_dpp = fnder(p3a_pp,1);
	dp3a = fnval(p3a_dpp,r);

	% Nonlocal pseudopotential for spin-orbit coupling and their
	% derivatives
	p1ka_dpp = fnder(p1ka_pp,1);
	dp1ka = fnval(p1ka_dpp,r);

	p1kb_dpp = fnder(p1kb_pp,1);
	dp1kb = fnval(p1kb_dpp,r);

	p1kc_dpp = fnder(p1kc_pp,1);
	dp1kc = fnval(p1kc_dpp,r);

	p2ka_dpp = fnder(p2ka_pp,1);
	dp2ka = fnval(p2ka_dpp,r);

	p2kb_dpp = fnder(p2kb_pp,1);
	dp2kb = fnval(p2kb_dpp,r);

	p2kc_dpp = fnder(p2kc_pp,1);
	dp2kc = fnval(p2kc_dpp,r);

	p3ka_dpp = fnder(p3ka_pp,1);
	dp3ka = fnval(p3ka_dpp,r);

	gd = find(r>0);
	Es = 1/2* sum(4*pi*r(gd).^2 .* Vloc(gd) .* rho(gd) * stp);

	cur = [Znuc mass Zion Es];

	spl = zeros(numel(r), 0);
	wgt = zeros(1,0);
	typ = zeros(1,0);
	cut = zeros(1,0);

	cnt = 1;


	spl(:,cnt) = r(:);
	wgt(cnt) = -1;    typ(cnt) =  9;    cut(cnt) = rhocut;    cnt=cnt+1;

	spl(:,cnt) = -rho(:);
	wgt(cnt) = -1;   typ(cnt)  =  99;    cut(cnt) = rhocut;    cnt=cnt+1;
	spl(:,cnt) = -drho(:);
	wgt(cnt) = -1;   typ(cnt)  =  99;  cut(cnt) = rhocut; cnt=cnt+1;

	if(abs(h0a) > 1e-10)
		spl(:,cnt) = p0a(:);
		wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp0a(:);
		wgt(cnt) = h0a;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h0b) > 1e-10)
		spl(:,cnt) = p0b(:);
		wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp0b(:);
		wgt(cnt) = h0b;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h0c) > 1e-10)
		spl(:,cnt) = p0c(:);
		wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp0c(:);
		wgt(cnt) = h0c;   typ(cnt) = 0;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h1a) > 1e-10)
		spl(:,cnt) = p1a(:);
		wgt(cnt) = h1a;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1a(:);
		wgt(cnt) = h1a;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h1b) > 1e-10)
		spl(:,cnt) = p1b(:);
		wgt(cnt) = h1b;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1b(:);
		wgt(cnt) = h1b;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h1c) > 1e-10)
		spl(:,cnt) = p1c(:);
		wgt(cnt) = h1c;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1c(:);
		wgt(cnt) = h1c;   typ(cnt) = 1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h2a) > 1e-10)
		spl(:,cnt) = p2a(:);
		wgt(cnt) = h2a;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2a(:);
		wgt(cnt) = h2a;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h2b) > 1e-10)
		spl(:,cnt) = p2b(:);
		wgt(cnt) = h2b;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2b(:);
		wgt(cnt) = h2b;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h2c) > 1e-10)
		spl(:,cnt) = p2c(:);
		wgt(cnt) = h2c;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2c(:);
		wgt(cnt) = h2c;   typ(cnt) = 2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(h3a) > 1e-10)
		spl(:,cnt) = p3a(:);
		wgt(cnt) = h3a;   typ(cnt) = 3;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp3a(:);
		wgt(cnt) = h3a;   typ(cnt) = 3;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k1a) > 1e-10)
		spl(:,cnt) = p1ka(:);
		wgt(cnt) = k1a;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1ka(:);
		wgt(cnt) = k1a;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k1b) > 1e-10)
		spl(:,cnt) = p1kb(:);
		wgt(cnt) = k1b;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1kb(:);
		wgt(cnt) = k1b;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k1c) > 1e-10)
		spl(:,cnt) = p1kc(:);
		wgt(cnt) = k1c;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp1kc(:);
		wgt(cnt) = k1c;   typ(cnt) = -1;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k2a) > 1e-10)
		spl(:,cnt) = p2ka(:);
		wgt(cnt) = k2a;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2ka(:);
		wgt(cnt) = k2a;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k2b) > 1e-10)
		spl(:,cnt) = p2kb(:);
		wgt(cnt) = k2b;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2kb(:);
		wgt(cnt) = k2b;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k2c) > 1e-10)
		spl(:,cnt) = p2kc(:);
		wgt(cnt) = k2c;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp2kc(:);
		wgt(cnt) = k2c;   typ(cnt) = -2;  cut(cnt) = wavcut; cnt=cnt+1;
	end

	if(abs(k3a) > 1e-10)
		spl(:,cnt) = p3ka(:);
		wgt(cnt) = k3a;   typ(cnt) = -3;  cut(cnt) = wavcut; cnt=cnt+1;
		spl(:,cnt) = dp3ka(:);
		wgt(cnt) = k3a;   typ(cnt) = -3;  cut(cnt) = wavcut; cnt=cnt+1;
	end


	if(1)
		fprintf('Check for normalization condition for element %3d\n', Znuc);
		fprintf('Total number of pseudopotentials terms %3d\n', cnt-1);

		% Local pseudopotential
		pk = find(r>0 & r<rhocut); 
		fprintf('int rho         = %15.10e\n', sum(4*pi*r(pk).^2.*(-rho(pk))*stp) );

		% Nonlocal pseudopotential
		pk = find(r>0 & r<wavcut);
		if( abs(h0a) > 1e-10 )
			fprintf('Abs: int p0a^2  = %15.10e\n', sum(r(pk).^2.*p0a(pk).^2*stp) );
			fprintf('Rel: int p0a^2  = %15.10e\n', ...
				sum(r(pk).^2.*p0a(pk).^2*stp) / (sum(r.^2.*p0a.^2*stp)/2) );
		end
		if( abs(h0b) > 1e-10 )
			fprintf('Abs: int p0b^2  = %15.10e\n', sum(r(pk).^2.*p0b(pk).^2*stp) );
			fprintf('Rel: int p0b^2  = %15.10e\n', ...
				sum(r(pk).^2.*p0b(pk).^2*stp) / (sum(r.^2.*p0b.^2*stp)/2) );
		end
		if( abs(h0c) > 1e-10 )
			fprintf('Abs: int p0c^2  = %15.10e\n', sum(r(pk).^2.*p0c(pk).^2*stp) );
			fprintf('Rel: int p0c^2  = %15.10e\n', ...
				sum(r(pk).^2.*p0c(pk).^2*stp) / (sum(r.^2.*p0c.^2*stp)/2) );
		end
		if( abs(h1a) > 1e-10 )
			fprintf('Abs: int p1a^2  = %15.10e\n', sum(r(pk).^2.*p1a(pk).^2*stp) );
			fprintf('Rel: int p1a^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1a(pk).^2*stp) / (sum(r.^2.*p1a.^2*stp)/2) );
		end
		if( abs(h1b) > 1e-10 )
			fprintf('Abs: int p1b^2  = %15.10e\n', sum(r(pk).^2.*p1b(pk).^2*stp) );
			fprintf('Rel: int p1b^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1b(pk).^2*stp) / (sum(r.^2.*p1b.^2*stp)/2) );
		end
		if( abs(h1c) > 1e-10 )
			fprintf('Abs: int p1c^2  = %15.10e\n', sum(r(pk).^2.*p1c(pk).^2*stp) );
			fprintf('Rel: int p1c^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1c(pk).^2*stp) / (sum(r.^2.*p1c.^2*stp)/2) );
		end
		if( abs(h2a) > 1e-10 )
			fprintf('Abs: int p2a^2  = %15.10e\n', sum(r(pk).^2.*p2a(pk).^2*stp) );
			fprintf('Rel: int p2a^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2a(pk).^2*stp) / (sum(r.^2.*p2a.^2*stp)/2) );
		end
		if( abs(h2b) > 1e-10 )
			fprintf('Abs: int p2b^2  = %15.10e\n', sum(r(pk).^2.*p2b(pk).^2*stp) );
			fprintf('Rel: int p2b^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2b(pk).^2*stp) / (sum(r.^2.*p2b.^2*stp)/2) );
		end
		if( abs(h2c) > 1e-10 )
			fprintf('Abs: int p2c^2  = %15.10e\n', sum(r(pk).^2.*p2c(pk).^2*stp) );
			fprintf('Rel: int p2c^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2c(pk).^2*stp) / (sum(r.^2.*p2c.^2*stp)/2) );
		end
		if( abs(h3a) > 1e-10 )
			fprintf('Abs: int p3a^2  = %15.10e\n', sum(r(pk).^2.*p3a(pk).^2*stp) );
			fprintf('Rel: int p3a^2  = %15.10e\n', ...
				sum(r(pk).^2.*p3a(pk).^2*stp) / (sum(r.^2.*p3a.^2*stp)/2) );
		end

		if( abs(k1a) > 1e-10 )
			fprintf('Abs: int p1ka^2  = %15.10e\n', sum(r(pk).^2.*p1ka(pk).^2*stp) );
			fprintf('Rel: int p1ka^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1ka(pk).^2*stp) / (sum(r.^2.*p1ka.^2*stp)/2) );
		end

		if( abs(k1b) > 1e-10 )
			fprintf('Abs: int p1kb^2  = %15.10e\n', sum(r(pk).^2.*p1kb(pk).^2*stp) );
			fprintf('Rel: int p1kb^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1kb(pk).^2*stp) / (sum(r.^2.*p1kb.^2*stp)/2) );
		end

		if( abs(k1c) > 1e-10 )
			fprintf('Abs: int p1kc^2  = %15.10e\n', sum(r(pk).^2.*p1kc(pk).^2*stp) );
			fprintf('Rel: int p1kc^2  = %15.10e\n', ...
				sum(r(pk).^2.*p1kc(pk).^2*stp) / (sum(r.^2.*p1kc.^2*stp)/2) );
		end

		if( abs(k2a) > 1e-10 )
			fprintf('Abs: int p2ka^2  = %15.10e\n', sum(r(pk).^2.*p2ka(pk).^2*stp) );
			fprintf('Rel: int p2ka^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2ka(pk).^2*stp) / (sum(r.^2.*p2ka.^2*stp)/2) );
		end

		if( abs(k2b) > 1e-10 )
			fprintf('Abs: int p2kb^2  = %15.10e\n', sum(r(pk).^2.*p2kb(pk).^2*stp) );
			fprintf('Rel: int p2kb^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2kb(pk).^2*stp) / (sum(r.^2.*p2kb.^2*stp)/2) );
		end

		if( abs(k2c) > 1e-10 )
			fprintf('Abs: int p2kc^2  = %15.10e\n', sum(r(pk).^2.*p2kc(pk).^2*stp) );
			fprintf('Rel: int p2kc^2  = %15.10e\n', ...
				sum(r(pk).^2.*p2kc(pk).^2*stp) / (sum(r.^2.*p2kc.^2*stp)/2) );
		end

		if( abs(k3a) > 1e-10 )
			fprintf('Abs: int p3ka^2  = %15.10e\n', sum(r(pk).^2.*p3ka(pk).^2*stp) );
			fprintf('Rel: int p3ka^2  = %15.10e\n', ...
				sum(r(pk).^2.*p3ka(pk).^2*stp) / (sum(r.^2.*p3ka.^2*stp)/2) );
		end

		fprintf('\n');
	end

	ess = {cur spl wgt typ cut};

	res{g,1} = Znuc;
	res{g,2} = ess;

end

if(1)
	binstr = sprintf('HGH.bin');
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

if(1)
	fid = fopen(binstr,'r');
	restst = deserialize(fid, string);
	fclose(fid);
end
