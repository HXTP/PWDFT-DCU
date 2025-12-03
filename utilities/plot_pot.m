%% Plot the potential obtained from the DGDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  This format is compatible with example/den2cube.cpp routine for more
%  complicated and efficient post processing
%
%  Lin Lin
%  Original: 2013/08/21
%  Revise:   2015/05/06
%  

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
numGridFineExtElem = round(numGridFineGlobal ./ numElem);
for d = 1 : 3
  if( numElem(d) > 1 )
    numGridFineExtElem(d) = numGridFineExtElem(d)*3;
  end
end
idxElem = 2;
fname = sprintf('POTEXT_%d', idxElem);
fid = fopen(fname, 'r');
key = deserialize( fid, {'Index3'} );
vtot = deserialize( fid, {'DblNumVec'});
vext = deserialize( fid, {'DblNumVec'});
fclose(fid);

vtot3D = reshape( vtot, numGridFineExtElem' );
vext3D = reshape( vext, numGridFineExtElem' );

d = 1;
xi = numGridFineExtElem(1)/2;
yi = numGridFineExtElem(2)/2;
zi = numGridFineExtElem(3)/3;

figure
hold on
if( d == 1 )
  plot(squeeze(vtot3D(:,yi,zi)), 'b-o');
  plot(squeeze(vext3D(:,yi,zi)), 'r-^');
  plot(squeeze(vtot3D(:,yi,zi)) - ...
    squeeze(vext3D(:,yi,zi)), 'k-d');
  title('x direction')
end
if( d == 2 )
  plot(squeeze(vtot3D(xi,:,zi)), 'b-o');
  plot(squeeze(vext3D(xi,:,zi)), 'r-^');
  plot(squeeze(vtot3D(xi,:,zi)) - ...
    squeeze(vext3D(xi,:,zi)), 'k-d');
  title('y direction')
end
if( d == 3 )
  plot(squeeze(vtot3D(xi,yi,:)), 'b-o');
  plot(squeeze(vext3D(xi,yi,:)), 'r-^');
  plot(squeeze(vtot3D(xi,yi,:)) - ...
    squeeze(vext3D(xi,yi,:)), 'k-d');
  title('z direction')
end
box on

hold off
legend('Vtot','Vext','Vorig');
