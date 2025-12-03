function egyCorr = pair_coulomb_correction( R, iType, jType )
% PAIR_COULOMB_CORRECTION computes the correction to the nuclei-nuclei
% interaction when the nuclei position are close to each other.
% 
%   egyCorr = pair_coulomb_correction( R, iType, jType ) gives the
%   correction for two atoms of type i and type j with distance being R.
%   The correction is given by the formula
%
%     egyCorr = 1/2 Z_I Z_J / |x-y| - 1/2 \int dx dy m_I(x) m_J(y) / |x-y| 
%   
%   where m_I, m_J are the pseudo-charge for the atom I and J.
%
% Lin Lin
% 08/24/2013

Znucs = union( iType, jType );
res = cell(length(Znucs), 3);
for g=1:length(Znucs)

	Znuc = Znucs(g); fprintf(1,'Znuc %d\n',Znuc);

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
	h022 = 0;
	h033 = 0;
	r1   = 1;
	h111 = 0;
	h122 = 0;
	h133 = 0;
	r2   = 1;
	h211 = 0;
	h222 = 0;
	h233 = 0;
	r3   = 1;
	h311 = 0;

	% Spin-orbit coupling
	k111 = 0;
	k122 = 0;
	k133 = 0;
	k211 = 0;
	k222 = 0;
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

	% H
	if(Znuc==1)
		Zion = 1;
		mass = 1.00794;
		rloc    = 0.200000;
		C1      = -4.180237;
		C2      = 0.725075;
		%
		rhocut  = 2.0;
		wavcut  = 2.0;
	end 

	% Li (semicore)
	if(Znuc==3)
		Zion = 3;
		mass = 6.941;
		rloc    = 0.400000;
		C1      = -14.034868;
		C2      = 9.553476;
		C3      = -1.766488;
		C4      = 0.084370;
		%
		rhocut  = 3.5;
		wavcut  = 3.5;
	end 

	% Li (no semicore)
	% if(Znuc==3)
		% Zion = 1;
		% mass = 6.941;
		% rloc    = 0.787553;
		% C1      = -1.892612;
		% C2      = 0.286060;
		% r0      = 0.666375;
		% h011    = 1.858811;
		% r1      = 1.079306;
		% h111    = -0.005895;
		% k111    = 0.000019;  
		% %
		% rhocut  = 7.5;
		% wavcut  = 7.5;
	% end 

	% C
	if(Znuc==6)
		Zion = 4;
		mass = 12.011;
		rloc = 0.348830;
		C1 = -8.513771;
		C2 = 1.228432;
		r0 = 0.304553;
		h011 = 9.522842;
		r1 = 0.232677;
		k111 = 0.004104;  
		rhocut = 3.0;
		wavcut = 3.0;
	end

	% O
	if(Znuc==8)
		Zion = 6;
		mass = 15.9994;
		rloc    = 0.247621;
		C1      = -16.580318;
		C2      = 2.395701;
		r0      = 0.221786;
		h011    = 18.266917;
		r1      = 0.256829;
		k111    = 0.004476;
		%
		rhocut  = 2.0;
		wavcut  = 2.0;
	end 


	% F
	if(Znuc==9)
		Zion = 7;
		mass = 18.9984032;
		rloc    = 0.218525;
		C1      = -21.307361;
		C2      = 3.072869;
		r0      = 0.195567;
		h011    = 23.584942;
		r1      = 0.174268;
		k111    = 0.015106;

		rhocut = 2.0;
		wavcut = 2.0;
	end 

	% Na (No semicore)
	if(Znuc==11)
		Zion = 1;
		mass = 22.989768;
		rloc = 0.885509;
		C1 = -1.238867;
		r0 = 0.661104;
		h011 = 1.847271;
		h022 = 0.582004;
		r1 = 0.857119;
		h111 = 0.471133;
		k111 = 0.002623;
		rhocut = 6.5;
		wavcut = 6.5;
	end

	% Al 
	if(Znuc==13)
		Zion = 3;
		mass = 26.9815386;
		rloc = 0.450000;
		C1   = -8.491351;
		r0   = 0.460104;
		h011 = 5.088340;
		h022 = 2.679700;
		r1   = 0.536744;
		h111 = 2.193438;
		k111 = 0.006154;
		k122 = 0.003947;
		rhocut = 3.5;
		wavcut = 3.5;
	end

	% Al: local potential only
	% if(Znuc==13)
		% Zion = 3;
		% mass = 26.9815386;
		% rloc = 0.450000;
		% C1   = -8.491351;
		% r0   = 0.460104;
		% rhocut = 3.5;
		% wavcut = 3.5;
	% end

	%Si
	if(Znuc==14)
		Zion = 4;
		mass = 28.0855;
		rloc = 0.440000;
		C1 = -7.336103;
		r0 = 0.422738;
		h011 = 5.906928;
		h022 = 3.258196;
		r1 = 0.484278;
		h111 = 2.727013;
		k111 = 0.000373;
		k122 = 0.014437;
		rhocut = 3.5;
		wavcut = 3.5;
	end

	% P
	if(Znuc==15) 
		Zion = 5;
		mass = 30.973762;
		rloc = 0.430000; 
		C1 = -6.654220;
		r0 = 0.389803;
		h011 = 6.842136;
		h022 = 3.856693;
		r1 = 0.440796;
		h111 = 3.282606;
		k111 = 0.002544;
		k122 = 0.017895;
		rhocut = 3.5;
		wavcut = 3.5;
	end
	

	% Se
	if(Znuc==34)
		Zion    = 6;
		mass    = 78.96;
		rloc    = 0.510000;
		r0      = 0.432531;
		h011    = 5.145131;
		h022    = 2.052009;
		h033    = -1.369203;
		r1      = 0.472473;
		h111    = 2.858806;
		h122    = -0.590671;
		k111    = 0.062196;
		k122    = 0.064907;
		r2      = 0.613420;
		h211    = 0.434829;
		k211    = 0.005784;
		rhocut = 3.0;
		wavcut = 3.0;
	end

	% Ce
	if(Znuc==58)
		Zion    = 12;
		mass    = 140.116;
		rloc    = 0.535000;
		C1      = 18.847470;
		C2      = -0.765636;
		r0      = 0.521790;
		h011    = 1.321616;
		h022    = -1.700444;
		r1      = 0.470324;
		h111    = 0.972641;
		h122    = -1.451337;
		h133    = 0.000000;
		k111    = 0.463710;
		k122    = 0.090257;
		k133    = 0.012566;
		r2      = 0.703593;
		h211    = 0.074241;
		k211    = 0.013265;
		r3      = 0.306717;
		h311    = -17.214790;
		k311    = 0.007568;
		%
		rhocut  = 3.5;
		wavcut  = 3.5;
	end 


	% Lu (WITH semi-core)

	if(Znuc==71)
		Zion = 25;
		mass = 174.9668;
		rloc    = 0.497000;
		C1      = 17.037053;
		C2      = -1.661610;
		r0      = 0.391206;
		h011    = 2.184678;
		h022    = -5.432346;
		r1      = 0.393896;
		h111    = -0.719819;
		h122    = -2.723799;
		k111    = 0.152450;
		k122    = 1.395416;
		k133    = -1.238744;
		r2      = 0.436518;
		h211    = -1.173245;
		k211    = 0.072130;
		r3      = 0.232629;
		h311    = -31.852262;
		k311    = 0.028006;
		%
		rhocut  = 3.5;
		wavcut  = 3.0;
	end 

	% Pt (WITHOUT Semi-core)
	if(Znuc==78 & 0)
		Zion    = 10;
		mass    = 195.084;
		rloc    = 0.616000;
		C1      = 11.027417;
		r0      = 0.520132;
		h011    = 2.447430;
		h022    = 2.640360;
		r1      = 0.658976;
		h111    = 0.408453;
		h122    = 1.647716;
		k111    = -0.763296;
		k122    = 1.065883;
		r2      = 0.451243;
		h211    = -4.552295;
		h222    = -2.102396;
		k211    = 0.146912;
		k222    = -0.169306;

		rhocut = 4.0;
		wavcut = 3.5;
	end


	% Pt (WITH Semi-core)
	if(Znuc==78)
		Zion    = 18;
		mass    = 195.084;
		rloc    = 0.500000;
		C1      = 5.445832;
		C2      = 1.156382;
		r0      = 0.409942;
		h011    = 2.994366;
		h022    = -7.448772;
		h033    = 4.243095;
		r1      = 0.398652;
		h111    = -0.225181;
		h122    = -3.776974;
		k111    = 1.017060;
		k122    = -0.348213;
		k133    = -0.331919;
		r2      = 0.367964;
		h211    = 0.632067;
		h222    = -5.755431;
		k211    = 0.226472;
		k222    = -0.114346;

		rhocut = 3.5;
		wavcut = 3.5;
	end


	% Au (WITH Semi-core)
	if(Znuc==79)
		Zion    = 11;
		mass    = 196.966569;
		rloc    = 0.590000;
		C1      = 11.604428;
		r0      = 0.521180;
		h011    = 2.538614;
		h022    = 2.701113;
		r1      = 0.630613;
		h111    = 0.394853;
		h122    = 2.057831;
		k111    = -0.960055;
		k122    = 1.296571;
		r2      = 0.440706; 
		h211    = -4.719070;
		h222    = -1.650429;
		k211    = 0.148484;
		k222    = -0.169493;

		rhocut = 3.5;
		wavcut = 3.5;
	end

	% Tl (WITH semi-core)
	if(Znuc==81)
		Zion    = 13;
		mass    = 204.3833;
		rloc    = 0.550000;
		C1      = 7.301886;
		r0      = 0.502423;
		h011    = 3.326560;
		h022    = 4.341390;
		r1      = 0.572016;
		h111    = 1.272807;
		h122    = 2.992206;
		k111    = 0.012233;
		k122    = 0.031664;
		k133    = 1.019164;
		r2      = 0.393185;
		h211    = -3.200652;
		h222    = -3.008296;
		k211    = 0.186849;
		k222    = -0.170651;

		rhocut = 3.5;
		wavcut = 3.5;
	end

	% Hg (WITH Semi-core)
	if(Znuc==80)
		Zion    = 12;
		mass    = 200.59;
		rloc    = 0.570000;
		C1      = 2.134572;
		r0      = 0.521802;
		h011    = 3.293920;
		h022    = 4.661001;
		r1      = 0.621648;
		h111    = 2.100960;
		h122    = 1.689988;
		k111    = 0.084989;
		k122    = 0.072771;
		k133    = 0.653348;
		r2      = 0.401894;
		h211    = -1.669886;
		h222    = -2.473265;
		k211    = 0.155759;
		k222    = -0.122282;

		rhocut = 3.5;
		wavcut = 3.5;
	end


	% Bi
	if(Znuc==83)
		Zion    = 5;
		mass    = 208.98040;
		rloc    = 0.605000;
		C1      = 6.679437;
		r0      = 0.678858;
		h011    = 1.377634;
		h022    = -0.513697;
		h033    = -0.471028;
		r1      = 0.798673;
		h111    = 0.655578;
		h122    = -0.402932;
		k111    = 0.305314;
		k122    = -0.023134;
		r2      = 0.934683;
		h211    = 0.378476;
		k211    = 0.029217;

		rhocut = 4.5;
		wavcut = 4.5;
	end


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

	res{g,1} = Zion;
	res{g,2} = Vlocfn;
	res{g,3} = rho0fn;
	res{g,4} = rhocut;

end


iType = find( Znucs == iType );
jType = find( Znucs == jType );
iZion = res{iType, 1};
jZion = res{jType, 1};
iVlocfn = res{iType, 2};
jVlocfn = res{jType, 2};
irho0fn = res{iType, 3};
jrho0fn = res{jType, 3};
irhocut = res{iType, 4};
jrhocut = res{jType, 4};
intRR = 2 * pi * ...
	dblquad(@(r,u)r.^2 .* irho0fn(r) .* ...
	jVlocfn(sqrt(r.^2 - 2 .*r .* R .* u + R^2)), ...
	0, irhocut, -1, 1,1e-11);
egyCorr = 0.5 * iZion * jZion / R - ...
	0.5 * intRR;
end
