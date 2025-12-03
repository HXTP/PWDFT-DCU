%% Plot the electron density obtained from the PWDFT calculation.
%
%  This is just a temporary code for post processing.  In future this
%  shall be addressed by cut3d like code.
%
%  Lin Lin
%  Revise:   2016/11/21

% First read the structural information
fname = 'DEN';
fid = fopen(fname, 'r');
for d = 1 : 3
  gridPos{d} = deserialize( fid, {'DblNumVec'} );
  numGridFine(d) = length(gridPos{d});
end
rho = deserialize( fid, {'DblNumVec'});
fclose(fid);

rho3D = reshape( rho, numGridFine );

d = 3;
% xi = numGridFine(1)/2;
% yi = numGridFine(2)/2;
xi = 1;
yi = 1;
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
