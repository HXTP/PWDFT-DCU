function pp = psp8read( pp, filename )
% PSP8READ  Parse an PSP8 format document into PpData type.
%    pp = PSP8READ(pp,filename) reads a file in the string input argument
%    filename.  The function assign the values in the file to the pp which
%    is of type PpData.
%
%
%    See also xmlread, PpData, upfread, upfreadold.

%  Copyright (c) 2016-2017 Yingzhou Li, Lin Lin and Chao Yang,
%                          Stanford University and Lawrence Berkeley
%                          National Laboratory
%  This file is distributed under the terms of the MIT License.

e2 = e2Def();

%-------------------------------------------------------------------------
% Read the entire file into a cell array
fid = fopen(filename);
strcell = textscan(fid,'%s','Delimiter','\n');
strcell = strcell{1};
strpt = 1;
fclose(fid);


%-------------------------------------------------------------------------
% Read the Header of psp8 file.

% line 1
psp.title = strline();

% line 2
psp = scanline(psp);

% line 3
psp = scanline(psp);
if psp.pspcod ~= 8
    error('psp version is not psp8');
end

% line 4
psp = scanline(psp);

% line 5
psp = scanline(psp,1,5);

% line 6
psp = scanline(psp,1,-1);

%-------------------------------------------------------------------------
% Set pp field with psp.
pp.info.pptype          = 'psp8';
pp.info.version         = psp.pspcod;
pp.info.generated       = psp.title;
pp.info.author          = '';
pp.info.date            = psp.pspd;
pp.info.comment         = '';
pp.info.element         = num2sym(psp.zatom);
pp.info.pseudo_type     = 'NC';
pp.info.relativistic    = 'nonrelativistic';
pp.info.functional      = pspxcfunctional(psp.pspxc);
pp.info.is_ultrasoft    = false;
pp.info.is_paw          = false;
pp.info.is_coulomb      = false;
pp.info.has_so          = false;
pp.info.has_wfc         = false;
pp.info.has_gipaw       = false;
pp.info.paw_as_gipaw    = false;
pp.info.core_correction = false;
pp.info.z_valence       = psp.zion;
pp.info.total_psenergy  = NaN;
pp.info.wfc_cutoff      = NaN;
pp.info.rho_cutoff      = psp.rchrg;
pp.info.l_max           = psp.lmax;
pp.info.l_max_rho       = NaN;
pp.info.l_local         = psp.lloc;
pp.info.mesh_size       = psp.mmax;
pp.info.number_of_wfc   = NaN;
pp.info.number_of_proj  = NaN;

pp.anum = psp.zatom;
pp.venum = psp.zion;


psp_mesh_size = psp.mmax;
psp_nbeta = sum(psp.nproj);

%-------------------------------------------------------------------------
% Read the PP_NONLOCAL of psp file.
if pp.info.l_max >= 0
    pp.nonloc.label                   = '';
    pp.nonloc.beta                    = zeros(psp_mesh_size,psp_nbeta);
    pp.nonloc.angular_momentum        = zeros(1,psp_nbeta);
    pp.nonloc.nbeta                   = psp_nbeta;
    dd = zeros(1,psp_nbeta);
    ibeta = 0;
    for itl = 0:pp.info.l_max
        betas = ibeta + (1:psp.nproj(itl+1));
        [pp.nonloc.beta(:,betas),dd(betas)] = ...
            pspreadnonloc(itl,psp_mesh_size,psp.nproj(itl+1));
        pp.nonloc.angular_momentum(betas) = itl;
        ibeta = ibeta + psp.nproj(itl+1);
    end
    pp.nonloc.lll                     = pp.nonloc.angular_momentum;
    pp.nonloc.dij                     = diag(dd)*e2;
    
    pp.nonloc.cutoff_radius_index     = NaN;
    pp.nonloc.cutoff_radius           = NaN;
    pp.nonloc.ultrasoft_cutoff_radius = NaN;
    pp.nonloc.norm_conserving_radius  = NaN;
end

%-------------------------------------------------------------------------
% Read the PP_LOCAL of psp file.
pp.vloc = pspreadloc(psp.lloc,psp_mesh_size)*e2;


%-------------------------------------------------------------------------
% Read the PP_MESH of psp file.
pp.info.dx     = NaN;
pp.info.mesh   = NaN;
pp.info.xmin   = NaN;
pp.info.rmax   = NaN;
pp.info.zmesh  = NaN;

% Read PP_R from PP_MESH
[pp.r,rhoc,pp.drhoc] = pspreadmesh(psp_mesh_size);
pp.rhoatom = rhoc.*pp.r.^2;

rab = pp.r(2:end)-pp.r(1:end-1);
pp.rab = [rab;mean(rab)];


%************************************************************************%
%                                                                        %
%                       Sub Functions in psp8read                        %
%                                                                        %
%************************************************************************%

%=========================================================================
% Subfunction strline
% Return the current line in strcell and move pointer to next line
    function str = strline()
        str = cell2val(strcell(strpt),'');
        strpt = strpt+1;
    end

%=========================================================================
% Subfunction cell2val
% Convert the first element of any recursive cell to value.
    function val = cell2val(c,val)
        if nargin < 2
            val = {};
        end
        if isempty(c)
            return;
        end
        if iscell(c)
            val = cell2val(c{1},val);
        else
            if ~isempty(c)
                val = c;
            end
        end
    end % end of subfunc cell2val
%TODO: move this subfunction to the outside as a tool

%=========================================================================
% Subfunction strline
% Return the current line in strcell and move pointer to next line
    function field = scanline(field,nattr,lenattr)
        strtmp = strline();
        c = textscan(strtmp,'%s','Delimiter',{' ',','}, ...
            'MultipleDelimsAsOne',1);
        c = c{1};
        c = strrep(c, 'D+', 'E+');
        c = strrep(c, 'D-', 'E-');
        if nargin == 1
            nattr = length(c)/2;
            lenattr = ones(1,nattr);
        elseif nargin == 2
            lenattr = ones(1,nattr);
        elseif nargin == 3 && nattr == 1 && lenattr(1) < 0
            lenattr = length(c)-1;
        end
        sn = sum(lenattr);
        offset = 0;
        for it = 1:nattr
            if strcmpi(cell2val(c{sn+it}),'pspd') && lenattr(it) == 1
                field.(cell2val(c{sn+it})) = cell2val(c{offset+1});
                offset = offset+1;
                continue;
            end
            if strcmpi(cell2val(c{sn+it}),'extension_switch')
                field.(cell2val(c{sn+it})) = cell2val(c{offset+1});
                offset = offset+1;
                continue;
            end
            field.(cell2val(c{sn+it})) = ...
                cell2mat( cellfun( @str2double, ...
                c(offset+(1:lenattr(it)))', ...
                'UniformOutput', false));
            offset = offset+lenattr(it);
        end
        if nattr == 0
            field = cell2mat(cellfun(@str2double,c', ...
                'UniformOutput', false));
        end
    end

%=========================================================================
% Subfunction pspreadnonloc
% Read table to psp content
    function strxc = pspxcfunctional(ixc)
        switch ixc
            case 0
                strxc = '';
            case 1
                strxc = '';
            case 2
                strxc = 'pz';
            case 3
                strxc = '';
            case 4
                strxc = '';
            case 5
                strxc = '';
            case 6
                strxc = '';
            case 7
                strxc = '';
            case 8
                strxc = '';
            case 9
                strxc = '';
            case 10
                strxc = '';
            case 11
                strxc = 'pbe';
            case 12
                strxc = '';
            case 13
                strxc = '';
            case 14
                strxc = '';
            case 15
                strxc = '';
            case 16
                strxc = '';
            case 17
                strxc = '';
            case 18
                strxc = '';
            case 19
                strxc = '';
            case 20
                strxc = '';
            case 21
                strxc = '';
            case 22
                strxc = '';
            case 23
                strxc = '';
            case 24
                strxc = '';
            case 25
                strxc = '';
            case 26
                strxc = '';
            case 27
                strxc = '';
            otherwise
                strxc = '';
        end
    end

%=========================================================================
% Subfunction pspreadnonloc
% Read table to psp content
    function [beta,ekb] = pspreadnonloc(ll,msize,nproj)
        c = scanline([],0);
        if c(1) ~= ll
            error('PSP file error in reading nonlocal pseudopotential');
        end
        ekb = c(2:end);
        
        beta = zeros(msize,nproj);
        for it = 1:msize
            c = scanline([],0);
            beta(it,:) = c(3:end);
        end
    end % end of subfunc pspreadnonloc

%=========================================================================
% Subfunction pspreadloc
% Read table to psp content
    function vloc = pspreadloc(lloc,msize)
        c = scanline([],0);
        if c(1) ~= lloc
            error('PSP file error in reading local pseudopotential');
        end
        
        vloc = zeros(msize,1);
        for it = 1:msize
            c = scanline([],0);
            vloc(it) = c(3);
        end
    end % end of subfunc pspreadloc

%=========================================================================
% Subfunction pspreadmesh
% Read table to psp content, higher order derivatives are not loaded.
    function [r,rhoc,drhoc] = pspreadmesh(msize)
        r      = zeros(msize,1);
        rhoc   = zeros(msize,1);
        drhoc  = zeros(msize,1);
        for it = 1:msize
            c = scanline([],0);
            r(it)      = c(2);
            rhoc(it)   = c(3);
            drhoc(it)  = c(4);
        end
    end % end of subfunc pspreadmesh

    %% The following functions are imported from KSSOLV


    function c = e2Def()
      % E2DEF  definition of the square of electron charge
      % 
      %    See also meDef.

      %  Copyright (c) 2015-2016 Yingzhou Li and Chao Yang,
      %                          Stanford University and Lawrence Berkeley
      %                          National Laboratory
      %  This file is distributed under the terms of the MIT License.

      % Hartree
      c = 1;

      % Rydberg
      % c = 2;

    end


    function symbol = num2sym(anum)
      % NUM2SYM  returns the atom symbol of the given atom number.
      %   symbol = NUM2SYM(anum) returns the atom symbol of the given number.
      %
      %   Example:
      %
      %       symbol = num2sym(3)
      %
      %       symbol =
      % 
      %       Li
      %
      %
      %   See also sym2num.

      %  Copyright (c) 2016-2017 Yingzhou Li and Chao Yang,
      %                          Stanford University and Lawrence Berkeley
      %                          National Laboratory
      %  This file is distributed under the terms of the MIT License.
      switch anum
        case 1
          symbol = 'H';
        case 2
          symbol = 'He';
        case 3
          symbol = 'Li';
        case 4
          symbol = 'Be';
        case 5
          symbol = 'B';
        case 6
          symbol = 'C';
        case 7
          symbol = 'N';
        case 8
          symbol = 'O';
        case 9
          symbol = 'F';
        case 10
          symbol = 'Ne';
        case 11
          symbol = 'Na';
        case 12
          symbol = 'Mg';
        case 13
          symbol = 'Al';
        case 14
          symbol = 'Si';
        case 15
          symbol = 'P';
        case 16
          symbol = 'S';
        case 17
          symbol = 'Cl';
        case 18
          symbol = 'Ar';
        case 19
          symbol = 'K';
        case 20
          symbol = 'Ca';
        case 21
          symbol = 'Sc';
        case 22
          symbol = 'Ti';
        case 23
          symbol = 'V';
        case 24
          symbol = 'Cr';
        case 25
          symbol = 'Mn';
        case 26
          symbol = 'Fe';
        case 27
          symbol = 'Co';
        case 28
          symbol = 'Ni';
        case 29
          symbol = 'Cu';
        case 30
          symbol = 'Zn';
        case 31
          symbol = 'Ga';
        case 32
          symbol = 'Ge';
        case 33
          symbol = 'As';
        case 34
          symbol = 'Se';
        case 35
          symbol = 'Br';
        case 36
          symbol = 'Kr';
        case 37
          symbol = 'Rb';
        case 38
          symbol = 'Sr';
        case 39
          symbol = 'Y';
        case 40
          symbol = 'Zr';
        case 41
          symbol = 'Nb';
        case 42
          symbol = 'Mo';
        case 43
          symbol = 'Tc';
        case 44
          symbol = 'Ru';
        case 45
          symbol = 'Rh';
        case 46
          symbol = 'Pd';
        case 47
          symbol = 'Ag';
        case 48
          symbol = 'Cd';
        case 49
          symbol = 'In';
        case 50
          symbol = 'Sn';
        case 51
          symbol = 'Sb';
        case 52
          symbol = 'Te';
        case 53
          symbol = 'I';
        case 54
          symbol = 'Xe';
        case 55
          symbol = 'Cs';
        case 56
          symbol = 'Ba';
        case 57
          symbol = 'La';
        case 58
          symbol = 'Ce';
        case 59
          symbol = 'Pr';
        case 60
          symbol = 'Nd';
        case 61
          symbol = 'Pm';
        case 62
          symbol = 'Sm';
        case 63
          symbol = 'Eu';
        case 64
          symbol = 'Gd';
        case 65
          symbol = 'Tb';
        case 66
          symbol = 'Dy';
        case 67
          symbol = 'Ho';
        case 68
          symbol = 'Er';
        case 69
          symbol = 'Tm';
        case 70
          symbol = 'Yb';
        case 71
          symbol = 'Lu';
        case 72
          symbol = 'Hf';
        case 73
          symbol = 'Ta';
        case 74
          symbol = 'W';
        case 75
          symbol = 'Re';
        case 76
          symbol = 'Os';
        case 77
          symbol = 'Ir';
        case 78
          symbol = 'Pt';
        case 79
          symbol = 'Au';
        case 80
          symbol = 'Hg';
        case 81
          symbol = 'Tl';
        case 82
          symbol = 'Pb';
        case 83
          symbol = 'Bi';
        case 84
          symbol = 'Po';
        case 85
          symbol = 'At';
        case 86
          symbol = 'Rn';
        case 87
          symbol = 'Fr';
        case 88
          symbol = 'Ra';
        case 89
          symbol = 'Ac';
        case 90
          symbol = 'Th';
        case 91
          symbol = 'Pa';
        case 92
          symbol = 'U';
        case 93
          symbol = 'Np';
        case 94
          symbol = 'Pu';
        case 95
          symbol = 'Am';
        case 96
          symbol = 'Cm';
        case 97
          symbol = 'Bk';
        case 98
          symbol = 'Cf';
        case 99
          symbol = 'Es';
        case 100
          symbol = 'Fm';
        case 101
          symbol = 'Md';
        case 102
          symbol = 'No';
        case 103
          symbol = 'Lr';
        case 104
          symbol = 'Rf';
        case 105
          symbol = 'Db';
        case 106
          symbol = 'Sg';
        case 107
          symbol = 'Bh';
        case 108
          symbol = 'Hs';
        case 109
          symbol = 'Mt';
        case 110
          symbol = 'Ds';
        case 111
          symbol = 'Rg';
        case 112
          symbol = 'Cn';
        case 113
          symbol = 'Uut';
        case 114
          symbol = 'Fl';
        case 115
          symbol = 'Uup';
        case 116
          symbol = 'Lv';
        case 117
          symbol = 'Uus';
        case 118
          symbol = 'Uuo';
        otherwise
          error('The input is not a supported atom number');
      end

    end


end % end of func upfread

