nreps         = [1 16 16];
asize         = [10.00 8.0540  4.65];
nat           = 4;
coefs = [
 0      0        0
 0      1/6      1/2
 0      1/2      1/2
 0      2/3      0
];
shift = [0.5 1/6 0.25];
coefs = coefs + repmat(shift, 4, 1);

gen_atompos(nreps, 'C', asize, nat, coefs, 0.0);
