ntyp          = 2;
nreps         = [1 2 2];
asize         = [4.138   7.167   28.640]; 
nat           = 30;
coefs = [
% Bi: A type
 0.000    0.000    0.399
 0.500    0.500    0.399
 0.000    0.000    0.601
 0.500    0.500    0.601
% Bi: B type
 0.000    0.333    0.066
 0.500    0.833    0.066
 0.000    0.333    0.267
 0.500    0.833    0.267
% Bi: C type
 0.000    0.667    0.732
 0.500    0.167    0.732
 0.000    0.667    0.934
 0.500    0.167    0.934
% Se: A type
 0.000    0.000    0.000
 0.500    0.500    0.000
 0.000    0.000    0.206
 0.500    0.500    0.206
 0.000    0.000    0.794
 0.500    0.500    0.794
% Se: B type
 0.000    0.333    0.460
 0.500    0.833    0.460
 0.000    0.333    0.667
 0.500    0.833    0.667
 0.000    0.333    0.873
 0.500    0.833    0.873
% Se: C type
 0.000    0.667    0.127
 0.500    0.167    0.127
 0.000    0.667    0.333
 0.500    0.167    0.333
 0.000    0.667    0.539
 0.500    0.167    0.539
];
coefs = coefs + 0.0;

typeList = [83; 34];
typeAtom  = [ones(12,1); 2*ones(18,1)];

typAtomVec = repmat(typeAtom, prod(nreps),1);

[C, xyzmat, xyzmatReduce] = GenReduceCoord(nreps, asize, nat, coefs, 0.0);

if(1)
  fh = fopen('rdc', 'w');
  for ityp = 1 : ntyp 
    fprintf(fh, 'Type %d\n\n', ityp);
    idx = find(typAtomVec == ityp);
    xyzmatReduceSel = xyzmatReduce(idx,:);
    for i = 1 : numel(idx)
      fprintf(fh, '%10.5f  %10.5f  %10.5f\n', ...
	xyzmatReduceSel(i,1), xyzmatReduceSel(i,2), ...
	xyzmatReduceSel(i,3));
    end
    fprintf(fh, '\n');
  end
end


% save('rdc','-ascii','xyzmatReduce');

% Print out cube file but with atomic structure only
if(1)
  fh = fopen('cub','w');
  fprintf(fh, 'Gaussian cube file for atomic structure only\n');
  fprintf(fh, 'Bi2Se3\n');
  numAtomTot = size(xyzmat,1);
  fprintf(fh, '%6d %12.5f %12.5f %12.5f\n', numAtomTot, 0, 0, 0);
  fprintf(fh, '%6d %12.5f %12.5f %12.5f\n', 1, C(1,1), 0, 0);
  fprintf(fh, '%6d %12.5f %12.5f %12.5f\n', 1, 0, C(2,2), 0);
  fprintf(fh, '%6d %12.5f %12.5f %12.5f\n', 1, 0, 0, C(3,3));
  for i = 1 : numAtomTot
    fprintf(fh,'%6d %12.5f %12.5f %12.5f %12.5f\n', ...
      typeList(typAtomVec(i)), 0,  ...
      xyzmat(i,1), xyzmat(i,2), xyzmat(i,3));
  end
  fclose(fh);
end
  

