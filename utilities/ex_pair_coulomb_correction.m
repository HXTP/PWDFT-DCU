%% ex_pair_coulomb_correction computes the total correction due to the
% pair Coulomb interaction for the nuclei-nuclei interaction.
% Input:
% atomPos:      atomic position
% atomType:     atom type
% domainLength: domain length

atomPos = ...
	[ +8.43280000e-02 +3.56520000e-01 +4.08849600e+00 
+3.90453600e+00 +4.24716000e+00 -3.56688000e-01 
+3.60477600e+00 +1.86720000e-01 +8.38502400e+00 
+4.14126400e+00 +3.86241600e+00 +7.81900800e+00 
+3.42038400e+00 +4.47282400e+00 +1.63181760e+01 
-4.88000000e-01 +3.81312000e+00 +1.99135920e+01 
-4.47424000e-01 +4.06104000e+00 +1.24992720e+01 
+3.45794400e+00 +4.24384000e-01 +1.19394480e+01 ];

atomType = [1 1 1 1 1 1 1 3]';

domainLength = [8.0 8.0 24.0]';

%% Compute the distance between atoms
numAtom = size(atomPos, 1);

dist = zeros(numAtom, numAtom);
for i = 1 : numAtom
	for j = 1 : numAtom
		td = atomPos(i,:)' - atomPos(j,:)';
		td = td - round(td ./ domainLength) .* domainLength;
		dist(i, j) = norm(td);
	end
end

egyCorr = zeros( numAtom, numAtom );
for i = 1 : numAtom
	for j = i+1 : numAtom
		egyCorr(i,j) = pair_coulomb_correction( dist(i,j), ...
			atomType(i), atomType(j) );
	end
end
egyCorr
disp('Total correction energy = ')
sum(egyCorr)

