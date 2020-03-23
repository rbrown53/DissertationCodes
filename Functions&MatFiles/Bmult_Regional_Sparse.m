  function y_array = Bmult_Regional4(x_array,params)
%  Compute array(y), where y = (A'*M*A+ alpha*P)*x. A is assumed BCCB, but 
%  we are using the data driven boundary conditions, thus a 
%  mask matrix M is required.  
% 
%  This one requires a permuation matrix, PermMat, for a sparse reordering.
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  nregions = params.nregions;
  R = params.R;
  PermMat = params.PermMat;
  DP = params.DP;
  D = params.D;
  ind_nz = params.ind_nz;
  [n,~] = size(x_array);
  N = n^2;
  
  % Compute P*x
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

%  Compute  A'*M*(A*x) + alpha P*x in array form
  y_array = real(ifft2(conj(ahat)...
      .*fft2(M.*real(ifft2(ahat.*fft2(x_array))))))...
      + alpha*reshape(Px,n,n);