nreps         = [2 2 2];
asize         = 7.9994; 
nat           = 2;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.5   
];
coefs = coefs + 0.0;

gen_atompos(nreps, 'Na', asize, nat, coefs, 0.0);
