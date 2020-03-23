function[Balpha] = Best_Alpha(alpha,c,params,Bmult_fn,x_true)
% This function evaluates the best alpha based on correlation with the true
% image.
% Inputs:
% alpha    = regularization parameter
% c        = A'*M'*b
% params   = structure array containing items needed to evaluate A
% Bmult_fn = function for evaluating A'*M'*M*A+\alpha*P.
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
params.precond_params  = 1./(abs(ahat).^2+alpha*phat);

% compute xalpha using CG
[nx,ny]                = size(ahat);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
xalpha                 = xalpha(nx/4+1:3*nx/4,nx/4+1:3*nx/4);

% Evaluate correaltion.
Balpha                 = -corr(xalpha(:),x_true(:));