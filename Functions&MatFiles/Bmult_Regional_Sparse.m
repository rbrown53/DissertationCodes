  function y_array = Bmult_Regional_Sparse(x_array,params)
%  Compute array(y), where y = (A'*M'*M*A+ alpha*P)*x. A is assumed BCCB, 
%  but we are using the data driven boundary conditions, thus a masking
%  array M is required.  
% 
%  This function requires a few extra objects stored in params. See section
%    5.1 of the dissertation for more information.
%  1. R: the cholesky factorization of P_{i,nz}.
%  2. PermMat: a permuation matrix for a sparse reordering.
%  3. D: a cell containing the masking matrices for each region.
%  4. DP: a cell containing D{i}*P{i} for i = 1,...,nregions
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  nregions = params.nregions;
  R = params.R;
  PermMat = params.PermMat;
  D = params.D;
  DP = params.DP;
  ind_nz = params.ind_nz;
  [n,~] = size(x_array);
  N = n^2;
  
  % Compute P*x following steps in Section 5.1.3 of dissertation
  Px = zeros(N,1);
  % More general for any non-overlapping regions.
  for region = 1:nregions
      PDx = DP{region}'*x_array(:); % P*D*x
      prodx = zeros(N,1);
      prodx(ind_nz{region}) = PermMat{region}'*(R{region}\(R{region}'\...
          (PermMat{region}*PDx(ind_nz{region})))); % pinv(~D*P*~D)*PDx
      prodx = DP{region}*prodx; % DP*pinv(~D*P*~D)*PDx
      Px = Px + DP{region}*(D{region}*x_array(:)) - prodx;
  end

%  Compute  A'*M'*M*(A*x) + alpha P*x in array form
  y_array = real(ifft2(conj(ahat)...
      .*fft2(M.*real(ifft2(ahat.*fft2(x_array))))))...
      + alpha*reshape(Px,n,n);