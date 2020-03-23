function[Balpha] = Best_Alpha_Regional(alpha,c,params,Bmult_fn,x_true)
% This function evaluates the best alpha based on correlation with the true
%   image in the regional case. Bmult_fn should be "Bmult_Regional_Sparse".
%   Used in the Inpainting_Regional_Sparse.m file.
% Inputs:
% alpha    = regularization parameter
% c        = A'*M'*b
% params   = structure array containing items needed to evaluate A
% Bmult_fn = function for evaluating A'*A+\alpha*P.
%
% Outputs:
% Balpha   = best alpha chosen by maximizing correlation (or minimizing
%               negative correlation).
% 
% written by Rick Brown 2019.
%
% extract a_params from params structure and add alpha for use within CG
a_params               = params.a_params;
ahat                   = a_params.ahat;
phat                   = a_params.phat;
a_params.alpha         = alpha;
params.a_params        = a_params;
% The following preconditioner using the average of the phat regions.
params.precond_params  = 1./(abs(ahat).^2+alpha*mean(cat(3,phat{:}),3));

% compute xalpha using CG, then compute Axalpha
[nx,ny]                = size(ahat);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
xalpha                = xalpha(nx/4+1:3*nx/4,nx/4+1:3*nx/4);


% Evaluate correlation.
Balpha               = -corr(xalpha(:),x_true(:));