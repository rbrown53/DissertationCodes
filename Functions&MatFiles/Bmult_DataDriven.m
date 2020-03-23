  function y_array = Bmult_DataDriven(x_array,params)
%  Compute array(y), where y = (A'*M'*M*A+ alpha*I)*x. A is assumed BCCB, but 
%  we are using the data driven boundary conditions, thus a masking
%  array M is required. For use when computing Tikhonov solution.
% 
%  written by John Bardsley 2016.
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  
  %  Compute lambda A'*M'*M*(A*x) + alpha L*x 
  
  dftx_array  = fft2(x_array);
  MAx         = M.*real(ifft2(ahat.*dftx_array));
  AtMAx_array = real(ifft2(conj(ahat).*fft2(MAx)));
  y_array     = AtMAx_array + alpha*x_array;
  
  