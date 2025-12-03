classdef PpData
    % PPDATA  Class for storing pseudopotential data.
    %    pp = PPDATA(ATOM) reads the pseudopotential provided by users,
    %    which should be in the path 'RootToKSSOLV/ppdata', if user
    %    do not provide the file, it will load the default pseudopotential
    %    file provided by KSSOLV corresponding to the ATOM, which can also
    %    be found in the path 'RootToKSSOLV/ppdata/default'.
    %
    %    pp = PPDATA(asym) reads the pseudopotential corresponding to the
    %    atom symbol asym.
    %
    %    pp = PPDATA(num) reads the pseudopotential corresponding to the
    %    atom number num.
    %
    %    Note: Currently only UPF files are supported.
    %
    %      >> pp = PpData('H')
    %
    %         pp = 
    % 
    %           PpData with properties:
    % 
    %                  anum: 1
    %                  info: [1x1 struct]
    %                 venum: 1
    %                     r: [602x1 double]
    %                   rab: [602x1 double]
    %                  vloc: [602x1 double]
    %               rho_atc: []
    %             semilocal: []
    %                nonloc: [1x1 struct]
    %               rhoatom: [602x1 double]
    %
    %    See also upfread.
    %
    %    Reference page in Help browser
    %       doc PspData
    
    %  Copyright (c) 2016-2017 Yingzhou Li and Chao Yang,
    %                          Stanford University and Lawrence Berkeley
    %                          National Laboratory
    %  This file is distributed under the terms of the MIT License.
    properties
        anum
        info
        venum
        r
        rab
        vloc
        rho_atc
        semilocal
        nonloc
        rhoatom
    end
    methods
        function pp = PpData(varargin)
            
            if nargin < 1
                return;
            end
            
            if ischar(varargin{1})
                ppfile = varargin{1};
                if nargin == 2
                    pptype = lower(varargin{2});
                else
                    pptype = 'upf';
                end
            else
                if isa(varargin{1},'Atom')
                    atom = varargin{1};
                    num = atom.anum;
                    asym = num2sym(num);
                elseif isnumeric(varargin{1})
                    num = varargin{1};
                    asym = num2sym(num);
                end
                
                ppregexp = [asym '[._-]' kssolvpptype '[\w.-]*(UPF|upf)'];
                pptype = 'upf';
                pppath = [kssolvroot 'ppdata/'];
                ppfile = dirregexp(pppath,ppregexp);
                if isempty(ppfile) || strcmpi(kssolvpptype,'default')
                    pppath = [kssolvroot 'ppdata/default/'];
                    ppregexp = [asym '[._-][\w.-]*(UPF|upf)'];
                    ppfile = dirregexp(pppath,ppregexp);
                end
                if isempty(ppfile)
                    error('The pseudopotential file does not exist.');
                end
            end
            
            switch pptype
                case 'upf'
                    pp = upfread(pp,ppfile);
                otherwise
                    error(['The pseudopotential file type'...
                           ' is not supported.']);
            end
            
            fprintf('The pseudopotential for %s is loaded from %s\n',...
                    num2sym(pp.anum),ppfile);
            
        end
    end
end