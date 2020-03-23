  function y_array = Bmult_DataDrivenMCMC(x_array,params)
%  Compute array(y), where y = (A'*M*A+ alpha*P)*x. A is assumed BCCB, but 
%  we are using the data driven boundary conditions, thus a 
%  mask matrix M is required.  
% 
%  written by John Bardsley 2016.
%

  ahat   = params.ahat;
  M      = params.M;
  lambda = params.lambda;
  delta  = params.delta;
  phat   = params.phat;
  
  %  Compute lambda A'*M*(A*x) + alpha L*x 
  
  dftx_array  = fft2(x_array);
  MAx         = M.*real(ifft2(ahat.*dftx_array));
  AtMAx_array = real(ifft2(conj(ahat).*fft2(MAx)));
  y_array     = lambda*AtMAx_array + delta*real(ifft2(phat.*dftx_array));
  