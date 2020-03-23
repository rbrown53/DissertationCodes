function [prior_samp, w] = prior_sample(masks,vars,extended)
% Sample from the prior for regions determined by masks and precision
% matrices defined by vars. vars holds l1, l2, theta, nu, the
% ratio (called tau in the paper), and a flag (0 or 1) that indicates
% isotropy or anisotropy, respectively.
% Alternatively, phat for each region, given as a cell, may be given
% instead of vars. P is assumed to have periodic boundary conditions.
% Extended is a flag. 1 means the masks given are defined on the extended
% domain, so the sample will be returned half the size.
% The sample is returned with a variance of 1.
if nargin < 3
    extended = 1;
end
n = size(masks{1},1);
d = 2; % two dimensions
h = 1/n;
nregions = max(size(masks));
if size(vars,2) == 6
    % Construct phat if the parameter estimates are given.
    phat = cell(1,2);
    for region = 1:nregions
        if vars(region,6) == 1
            l1 = vars(region,1);
            l2 = vars(region,2);
            theta = vars(region,3);
            nu = vars(region,4);
            at = l2*sin(theta); % a_theta
            bt = l1*cos(theta); % b_theta
            ct = l2*cos(theta); % c_theta
            dt = l1*sin(theta); % d_theta
            lx = zeros(n,n); lx(1,1) = 2; lx(1,2) = -1; lx(1,n) = -1;
            ly = zeros(n,n); ly(1,1) = 2; ly(2,1) = -1; ly(n,1) = -1;
            lxy = zeros(n,n); lxy(2,2) = 1;lxy(2,n) = -1;
            lxy(n,2) = -1; lxy(n,n) = 1;
            lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
               -2*(at*ct-bt*dt)/(4*h^2)*lxy);
            phat{region} = real((lhat + 1).^(nu + d/2));
        else
            ell = vars(region,1);
            nu = vars(region,4);
            l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
            l(n,1) = -1; l(1,n) = -1;
            lhat = fft2(ell^2/h^2*l);
            phat{region} = real((lhat + 1).^(nu + d/2));
        end
    end
else
    phat = vars;
end

%% Obtain the weights so each region has the same variance.
w = prior_weights(phat,masks);

%% Obtain the sample
X = zeros(n,n,100);
for i = 1:100
    E1 = randn(n,n);
    E2 = randn(n,n);
    for region = 1:nregions
        X(:,:,i) = X(:,:,i) + masks{region}.*real(ifft2((w(region)*phat{region}).^(-1/2)...
            .*fft2(masks{region}.*(E1+sqrt(-1)*E2))));
    end
end
X = mean(X,3);
if extended == 1
    prior_samp = X(n/4+1:3*n/4,n/4+1:3*n/4);
else
    prior_samp = X;
end
prior_samp = prior_samp/std(prior_samp(:));
imagesc(prior_samp),colorbar