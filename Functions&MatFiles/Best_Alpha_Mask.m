function[Balpha] = Best_Alpha_Mask(alpha,c,params,Bmult_fn,x_true,mask)
% This function evaluates the best alpha based on correlation with the true
%   image. This function is used when masks are needed to limit x_true to
%   only the region in which we are solving the inverse problem. Used in
%   the Inpainting_Mult.m file.
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
% written by John Bardsley 2016.
%
% extract a_params from params structure and add alpha for use within CG
a_params               = params.a_params;
ahat                   = a_params.ahat;
phat                   = a_params.phat;
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(ahat).^2+alpha*phat);

mask = double(mask);
mask(mask == 0) = NaN;

% compute xalpha using CG
[nx,ny]                = size(ahat);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
% Limit xalpha to only the region in question
xalpha                = xalpha(nx/4+1:3*nx/4,nx/4+1:3*nx/4).*mask;
xalphavec = xalpha(:);
xalphavec = rmmissing(xalphavec);
x_trueMask = x_true.*mask;
x_trueMaskvec = x_trueMask(:);
x_trueMaskvec = rmmissing(x_trueMaskvec);

Balpha               = -corr(xalphavec,x_trueMaskvec);