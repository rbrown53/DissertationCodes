  function y_array = Bmult_Regional(x_array,params)
%  Compute array(y), where y = (A'*M*A+ alpha*P)*x. A is assumed BCCB, but 
%  we are using the data driven boundary conditions, thus a 
%  mask matrix M is required.  
% 
%  written by John Bardsley 2016.
%

  ahat   = params.ahat;
  M      = params.M;
  alpha  = params.alpha;
  phat   = params.phat;
  nregions = params.nregions;
  Mask = params.Mask;
  
  %  Compute lambda A'*M*(A*x) + alpha L*x 
  
  dftx_array  = fft2(x_array);
  MAx         = M.*real(ifft2(ahat.*dftx_array));
  AtMAx_array = real(ifft2(conj(ahat).*fft2(MAx)));
  
  y_array     = AtMAx_array;
  for i = 1:nregions
      y_array = y_array + alpha*Mask{i}.*real(ifft2(phat{i}.*fft2(Mask{i}.*x_array))); % Equivalent to alpha*D'*P{i}*D*x
  end