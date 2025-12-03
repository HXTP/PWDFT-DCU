nreps         = [1 1 2];
asize         = 6.8311;
nat           = 4;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.0   
 0.5      0.0      0.5   
 0.0      0.5      0.5   
];
shift = [0.25 0.25 0.25];  % no shift
coefs = coefs + repmat(shift, 4, 1);

gen_atompos(nreps, 'Cu', asize, nat, coefs, 0.0);
