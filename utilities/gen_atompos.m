function gen_atompos(nreps, atype, asize, nat, coefs, ran, suffix);
% Generate the atomic position for a give crystal structure with
% potential randomness

au2ang = 0.52917721;
nrepx = nreps(1); nrepy = nreps(2); nrepz = nreps(3);

fprintf('Constructing %5s cells with size %3d*%3d*%3d\n', atype, ...
  nreps(1), nreps(2), nreps(3));
fprintf('Total number of atoms:   %6d\n\n', nat * prod(nreps));

if( numel(asize) == 1 )
  asize = ones(3,1)*asize;
elseif( numel(asize) == 3 )
  % do nothing
else
  error('asize is not in the correct form');
end
C = diag(asize);

xyzmat = coefs*C';
%
% repeat the cell nrep times along the xyz directions
%
for krep = 1 : nrepz
  for jrep = 1 : nrepy
    for irep = 1 : nrepx
      if( irep ~= 1 || jrep ~= 1 || krep ~= 1 ) 
        xyzpick = xyzmat(1:nat,:);
        xyzpick(:,1) = xyzpick(:,1) + (irep-1)*asize(1);
        xyzpick(:,2) = xyzpick(:,2) + (jrep-1)*asize(2);
        xyzpick(:,3) = xyzpick(:,3) + (krep-1)*asize(3);
        xyzmat = [xyzmat; xyzpick];
      end
    end
  end
end
% Add randomness
xyzmat = xyzmat + ran * (randn(size(xyzmat)));
%
% modify the supercell
%
C(1,1) = nrepx*asize(1);
C(2,2) = nrepy*asize(2);
C(3,3) = nrepz*asize(3);

fprintf('C\n%15.6f    %15.6f    %15.6f\n', C(1,1), C(2,2), C(3,3));
fprintf('xyzmat\n');
fprintf('%15.6f    %15.6f    %15.6f\n', xyzmat'); % Transpose is important!

%
% 4. Configure the molecule (crystal)
%
fh = fopen('atompos','w');

fprintf(fh, 'begin Super_Cell\n', 1);
fprintf(fh, '%12.6f     %12.6f    %12.6f\n', C(1,1), C(2,2), C(3,3));
fprintf(fh, 'end Super_Cell\n\n', 1);

fprintf(fh, 'Atom_Types_Num:   %6d\n\n', 1);
fprintf(fh, 'Atom_Type:        %6s\n\n', atype);

fprintf(fh, 'begin Atom_Coord\n', 1);
fprintf(fh, '%12.6f     %12.6f    %12.6f\n', xyzmat');
fprintf(fh, 'end Atom_Coord\n\n', 1);

fclose(fh);

fprintf('\n Done. Generated atompos file.\n');

if(1)
  disp('Reduced coordinates')
  rdcmat = xyzmat ./ repmat([C(1,1), C(2,2), C(3,3)], ...
    size(xyzmat,1), 1);
  if( nargin <= 6 ) 
    filename = 'rdc';
  else
    filename = sprintf( 'rdc_%s', suffix );
  end 
  save(filename,'-ascii','rdcmat');
end

if(1)
   
  disp('PWDFT/DGDFT input file');
  
  rdcmat = xyzmat ./ repmat([C(1,1), C(2,2), C(3,3)], ...
    size(xyzmat,1), 1);

  if( nargin <= 6 ) 
    filename = 'pwdft_dgdft_input';
  else
    filename = sprintf( 'dgdft_pwdft_input_%s', suffix );
  end
  
 fh = fopen(filename,'w'); 
  
 fprintf(fh, 'begin Super_Cell\n', 1);
 fprintf(fh, '%12.6f     %12.6f    %12.6f\n', C(1,1), C(2,2), C(3,3));
 fprintf(fh, 'end Super_Cell\n\n', 1);

 fprintf(fh, 'Atom_Types_Num:   %6d\n\n', 1);
 fprintf(fh, 'Atom_Type:        %6s\n\n', atype);

 fprintf(fh, 'begin Atom_Red\n', 1);
 fprintf(fh, '%12.6f     %12.6f    %12.6f\n', rdcmat');
 fprintf(fh, 'end Atom_Red\n\n', 1);

 fclose(fh);
  
end



if(1)
  disp('Lattice coordinates')
  latmat = rdcmat .* repmat([nreps(1), nreps(2), nreps(3)], ...
    size(xyzmat,1), 1);
  if( nargin <= 6 ) 
    filename = 'lattice';
  else
    filename = sprintf( 'lattice_%s', suffix );
  end
  save(filename,'-ascii','latmat');
end


if(1)
  disp('Cell size in angstrom')
  Cang = C * au2ang
  disp('Coordinate in angstrom')
  angmat = xyzmat * au2ang;
  if( nargin <= 6 ) 
    filename = 'angstrom';
  else
    filename = sprintf( 'angstrom_%s', suffix );
  end
  save(filename,'-ascii','Cang', 'angmat');
end


if(1)
    angmat = xyzmat * au2ang;
    fname = sprintf('atompos_%s.xyz',atype);
    fid = fopen(fname, 'w');
    sz = length(angmat);
    fprintf(fid,'%d\n',sz);
    for ii=1:sz
    
        fprintf(fid, '\n%s\t%1.8f\t%1.8f\t%1.8f', atype, angmat(ii,1), ...
            angmat(ii,2), angmat(ii, 3));
        
    end
    
    fclose(fid);
end
