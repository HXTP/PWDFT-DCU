%% Plot the electron density obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  Lin Lin
%  Original: 2013/08/15
%  Revise:   2015/05/06

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

% Look at the potential for a given element
idxElem = 2;
fname = sprintf('DEN_%d', idxElem);
fid = fopen(fname, 'r');
for d = 1 : 3
  gridPos{d} = deserialize( fid, {'DblNumVec'} );
  numGridFine(d) = length(gridPos{d});
end
key = deserialize( fid, {'Index3'} );
rho = deserialize( fid, {'DblNumVec'});
fclose(fid);

rho3D = reshape( rho, numGridFineExtElem' );

d = 1;
xi = numGridFine(1)/2;
yi = numGridFine(2)/2;
zi = 1;

figure
hold on
if( d == 1 )
  plot(gridPos{1},squeeze(rho3D(:,yi,zi)), 'b-o');
  title('x direction')
end
if( d == 2 )
  plot(gridPos{2},squeeze(rho3D(xi,:,zi)), 'b-o');
  title('y direction')
end
if( d == 3 )
  plot(gridPos{3},squeeze(rho3D(xi,yi,:)), 'b-o');
  title('z direction')
end
box on

hold off
legend('rho');
