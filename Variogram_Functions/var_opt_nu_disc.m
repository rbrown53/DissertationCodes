function [bestnu,obj_func] = var_opt_nu_disc(d,params,range0,sill0,nugget0)
% This function optimizes the variable nu discretely so nu can only be an
% integer. nu is chosen as the one that has the smallest weighted least
% squares value. 
% Inputs:
% d = dimension of the inverse problem 
% params = output of the variogram function
% range0 = best guess for the range parameter (distance at which the
%   semivariogram is at 95% of the sill)
% sill0 = best guess for the sill 
% nugget0 = best guess for the nugget

% Written by Rick Brown

% Determine the lower bound for nu
if d == 1
    lower = 1/2;
elseif d == 2
    lower = 1;
end
% Set the upper bound at 5. 
upper = 5;
obj_func = zeros(length(lower:1:upper),1);
nu = zeros(length(lower:1:upper),1);
k = 1;
for nu_opt = lower:1:upper
    [~,~,~,s]=variogramfit(params.distance,params.val,range0,sill0,params.num,'plotit',false,...
                'model','matern','nu',nu_opt,'weightfun','cressie85','nugget',nugget0);  
    obj_func(k) = sum(s.weights.*s.residuals.^2);
    nu(k) = nu_opt;
    k = k+1;
end
[~, nu_index] = min(round(obj_func/max(obj_func),1));
bestnu = nu(nu_index);