%% Plot the adaptive local basis functions obtained from the DGDFT
%calculation on a uniform grid.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  This format is compatible with example/den2cube.cpp routine for more
%  complicated and efficient post processing
%
%  Lin Lin
%  Original:   2015/05/07

% First read the structural information
fname = sprintf('STRUCTURE');
fid = fopen(fname, 'r');
domainSizeGlobal  = deserialize( fid, {'Point3'} );
numGridGlobal     = deserialize( fid, {'Index3'} );
numGridFineGlobal = deserialize( fid, {'Index3'} );
posStartGlobal    = deserialize( fid, {'Point3'} );
numElem           = deserialize( fid, {'Index3'} );
% Neglect the part for deserializing the atomList
fclose( fid );

% Look at the basis function for a given element
numGridUniform = numGridFineGlobal ./ numElem;

% Parameters
mpirank = 1;


fname = sprintf('ALBUNIFORM_%d', mpirank);
fid = fopen(fname, 'r');
% FIXME order is wrong here
key = deserialize( fid, {'Index3'} );
% FIXME grid was wrong here
gridPosTmp = cell(3,1);
for d = 1 : 3
  gridPosTmp{d} = deserialize( fid, {'DblNumVec'} );
end
wavefun = deserialize( fid, {'DblNumMat'} );
numWavefun = size(wavefun,2);
wavefun =  reshape( wavefun, [numGridUniform' numWavefun] );
fclose(fid);

d = 1;
xi = round(numGridUniform(1)/2);
yi = round(numGridUniform(2)/2);
zi = numGridUniform(3);
n  = 1;

figure
hold on
if( d == 1 )
  plot(squeeze(wavefun(:,yi,zi,n)), 'b-o');
  title('x direction')
end
if( d == 2 )
  plot(squeeze(wavefun(xi,:,zi,n)), 'b-o');
  title('y direction')
end
if( d == 3 )
  plot(squeeze(wavefun(xi,yi,:,n)), 'b-o');
  title('z direction')
end
box on
hold off
legend('Uniform fine')
