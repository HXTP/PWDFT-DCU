nreps         = [1  1  4];
asize         = 10.7;

% Ga
nat           = 4;
coefs = [
 0.0      0.0      0.0
 0.5      0.5      0.0   
 0.5      0.0      0.5  
 0.0      0.5      0.5 
];
coefs = coefs + 0.125;

gen_atompos(nreps, 'Ga', asize, nat, coefs, 0.05,'Ga');


% As
nat           = 4;
coefs = [
 0.25     0.25     0.25
 0.75     0.75     0.25
 0.75     0.25     0.75
 0.25     0.75     0.75 
];

coefs = coefs + 0.125;

gen_atompos(nreps, 'As', asize, nat, coefs, 0.05,'As');
