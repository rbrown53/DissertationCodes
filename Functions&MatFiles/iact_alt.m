function [tau, Kbar] = iact_alt(chain)
%IACT estimates the integrated autocorrelation time
%   using Sokal's adaptive truncated periodogram estimator:
%   $\hat{tau}_{int} = 1 + 2\sum_{j=1}^{Kbar} \frac{K}{K-j}\hat{rho}(j)$,
%   which is equivalent to
%   $\hat{tau}_{int} = \sum_{j=-Kbar}^{Kbar}\frac{K}{K-|j|}\hat{rho}(j)$
%   since \hat{rho}(j) = \hat{rho}(-j).
% For efficientcy, the chain should be one-dimensional. Use the iact.m
%   function for a multi-dimensional chain using FFTs.

K = length(chain);
A = acf(chain,K-1); % The acf function divides by K, not K-j.
A = A.*(K./(K-(0:K-1))); % Adjust to divide by K-j instead of K for j = 0:K-1.
for i = 1:K
    tau_test = 1 + 2*sum(A(2:i)); % Equivalent to sum([A(1:i) A(2:i)]);
    if i >= 3*tau_test % The suggested cutoff is 3*\hat{tau}.
        tau = tau_test;
        Kbar = i;
        break
    end
end