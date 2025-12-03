nreps         = [1 1 4];
asize         = 10.2160;
nat           = 8;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.0   
 0.5      0.0      0.5  
 0.0      0.5      0.5 
 0.25     0.25     0.25
 0.75     0.75     0.25
 0.75     0.25     0.75
 0.25     0.75     0.75 
];
coefs = coefs + 0.125;

gen_atompos(nreps, 'Si', asize, nat, coefs, 0.00);
