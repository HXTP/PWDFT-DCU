%% Plot the adaptive local basis functions obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  This format is compatible with example/den2cube.cpp routine for more
%  complicated and efficient post processing
%
%  Lin Lin
%  Original: 2013/08/16
%  Revise:   2015/05/07

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
numGridExtElem = round(numGridGlobal ./ numElem);
for d = 1 : 3
  if( numElem(d) > 1 )
    numGridExtElem(d) = numGridExtElem(d)*3;
  end
end

% Parameters
mpirank = 1;
d = 1;
xi = round(numGridExtElem(1)/2);
yi = round(numGridExtElem(2)/2);
zi = round(numGridExtElem(3)/3*2);
n  = 2;


fname = sprintf('WFNEXT_%d', mpirank );
fid = fopen(fname, 'r');
gridPos = cell(3,1);
for d = 1 : 3
  gridPos{d} = deserialize( fid, {'DblNumVec'} );
end
key = deserialize( fid, {'Index3'} );
wavefun = deserialize( fid, {'DblNumTns'} );
numWavefun = size(wavefun,3);
wavefun =  reshape( wavefun, [numGridExtElem' numWavefun] );
fclose(fid);


figure
hold on
if( d == 1 )
  plot(gridPos{1},squeeze(wavefun(:,yi,zi,n)), 'b-o');
  title('x direction')
end
if( d == 2 )
  plot(gridPos{2},squeeze(wavefun(xi,:,zi,n)), 'b-o');
  title('y direction')
end
if( d == 3 )
  plot(gridPos{3},squeeze(wavefun(xi,yi,:,n)), 'b-o');
  title('z direction')
end
box on
hold off
