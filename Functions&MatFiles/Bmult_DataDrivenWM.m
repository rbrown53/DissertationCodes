  function y_array = Bmult_DataDrivenWM(x_array,params)
%  Compute array(y), where y = (A'*M'*M*A+ alpha*P)*x. A is assumed BCCB,
%  but we are using the data driven boundary conditions, thus a masking
%  array M is required. WM stands for Whittle-Matern.
% 
%  written by Rick Brown 2018.
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  phat   = params.phat;
  
  %  Compute  A'*M'*M*(A*x) + alpha*P*x 
  
  dftx_array  = fft2(x_array);
  MAx         = M.*real(ifft2(ahat.*dftx_array));
  AtMAx_array = real(ifft2(conj(ahat).*fft2(MAx)));
  y_array     = AtMAx_array + alpha*real(ifft2(phat.*dftx_array));
  