  function y_array = Bmult_Regional(x_array,params)
%  Compute array(y), where y = (A'*M'*M*A+ alpha*P)*x. A is assumed BCCB,
%    but we are using the data driven boundary conditions, thus a masking
%    array M is required. Also, P is assume to be a cell with a different
%    precision matrix for each region. Mask is the masking matrix that
%    determine the different regions.
%  P*x = sum(D'*P{i}*D*x) for i = 1,...,nregions.
%
%  This function is used in Inpainting_Regional_Sparse.m to obtain an
%    approximate alpha value much quicker than using Bmult_Regional_Sparse.
% 
%  written by Rick Brown 2019.
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  phat   = params.phat;
  nregions = params.nregions;
  Mask = params.Mask;
  
  %  Compute lambda A'*M'*M*(A*x) + alpha*P*x 
  
  dftx_array  = fft2(x_array);
  MAx         = M.*real(ifft2(ahat.*dftx_array));
  AtMAx_array = real(ifft2(conj(ahat).*fft2(MAx)));
  
  y_array     = AtMAx_array;
  for i = 1:nregions
      % The following is equivalent to alpha*D'*P{i}*D*x
      y_array = y_array + alpha*Mask{i}.*real(ifft2(phat{i}.*fft2(Mask{i}.*x_array)));
  end