function ferr = maxforceerr(f1,f2,natom)
assert( mod(size(f1,1), natom) == 0 );
assert( mod(size(f2,1), natom) == 0 );

if( size(f1,1)==size(f2,1) )
    % Compare force for the same size
    ferr = f1-f2;
    ferr = sqrt(sum(ferr.^2,2));
    ferr = reshape(ferr,natom,[]);
    ferr = max(ferr,[], 1)';
elseif( size(f2,1) == natom )
    % Compare force against some benchmark result
    ferr = f1 - repmat(f2,size(f1,1)/natom,1);
    ferr = sqrt(sum(ferr.^2,2));
    ferr = reshape(ferr,natom,[]);
    ferr = max(ferr,[], 1)';
else
    error('Size of f2 should be either the same as f1 or natom');
end
        