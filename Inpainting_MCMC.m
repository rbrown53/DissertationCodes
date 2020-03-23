%  
%  Hierarchical Gibbs sampler for 2d image inpainting. Periodic boundary 
%  conditions are assumed on the image so that the DFT can be used for 
%  fast computations.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
%
%  The Geweke test is used to determine whether the second half of the 
%  chain is in equilibrium, and the integrated auto correlated time, 
%  and related essential sample size.
% 
%  Written by Rick Brown 2020 based on code written by John Bardsley 2016
%

clear all, close all
path(path,'Functions&MatFiles')
path(path,'Variogram_Functions')
path(path,'Example_Images')

%% Section 1 - Color, Prior, and Reading Data %%
% Make selections about saving plots, whether to do the deblurring in black
% and white vs color, and read in the data.
BW = input('Enter 1 to black and white deblur or 2 for color deblur ');

% Entering a number other than 1 will only perform MCMC for lambda and delta.
MCMCparam = input('Enter 1 to do MCMC for ell. ');
x_read = imread('Main_Hall_2008.jpg','jpg');
x_read = im2double(x_read);
[~,n,~] = size(x_read);
sig = 0.002; % Controls the amount of blur in the image.
mask_type = 1; % 1 for random mask, 2 for systematic mask
pmask = 0.4; % Percentage of elements that are removed from the image.

if BW == 1 
    x_true = rgb2gray(x_read);
    x_true = x_true/max(x_true(:))*100;
    coloriters = 1;
elseif BW == 2
    xmeancolor=zeros(n/2,n/2,3);
    bcolor=zeros(n/2,n/2,3);
    x_truecolor = zeros(n/2,n/2,3);
    Q = zeros(n/2,n/2,3);
    Lambda = cell(3,1);
    Delta = cell(3,1);
    if MCMCparam == 1
        Ell = cell(3,1);
    end
    coloriters = 3;
end

%% Section 2 - Generate Noisy Data %%
%  Construct the Fourier transform of the circulant-shifted
%  convolution kernel on the extended domain, then restrict the domain to 
%  generate the data.

% coloriters = 1 for black and white or = 3 for color. If dealing with
% color, we will essentially do 3 black and white deblurs and combine them
% into one color image since color images are stored as 3 2-D arrays - one
% each for the intensities of Red, Green, and Blue.

for coloriter = 1:coloriters
    if BW ==2
        x_true = x_read(:,:,coloriter);
    end
    % Generate data on 256^2 grid w/ periodic BCs, then restrict to 128^2.
    h      = 1/n;
    N      = n^2;
    x      = (-0.5+h/2:h:0.5-h/2)';
    [X,Y]  = meshgrid(x);
    kernel = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
    ahat   = fft2(fftshift(kernel));
    Ax     = real(ifft2(ahat.*fft2(x_true)));

    % Extract 128x128 middle subregion from Ax and x, add noise and plot.
    Ax     = Ax(n/4+1:3*n/4,n/4+1:3*n/4);
    x_true = x_true(n/4+1:3*n/4,n/4+1:3*n/4);

    [~,n2]= size(Ax);
    err_lev= 5;
    noise  = err_lev/100 * norm(Ax(:)) / sqrt(n2^2);
    rng(0) % fixes the seed of the random number generator.
    eta = noise*randn(n2,n2);
    b      = Ax + eta;
    if BW == 1
        figure(1), imagesc(x_true, [0,max(x_true(:))]), colormap(gray), colorbar, axis image
        yticks([9 29 49 69 89 109])
        yticklabels({'120', '100', '80', '60', '40', '20'})
        pause(.1);
    end
    xcoord = repelem((1:n2)',n2)/n2; % Assign an x coordinate to each b element
    ycoord = repmat((n2:-1:1)',n2,1)/n2; % Assign an y coordinate to each b element
    bdf = [xcoord ycoord b(:)];
    if mask_type == 1
        rng(0)
        bdf = [xcoord ycoord binornd(1,1-pmask,n2^2,1).*b(:)];
        b = reshape(bdf(:,3),n2,n2);
    elseif mask_type == 2
        Amask = eye(n2);
        for i=1:n2/4-1
            Amask(4*i:4*i+1,4*i:4*i+1)=zeros(2,2);
        end
        b = Amask*b*Amask;
        bdf = [xcoord ycoord b(:)];
    end
    % zero pad b and create the mask that matches the zeroed-out values in b.
    b_pad = padarray(b,[n/4,n/4]);
    b_pad(isnan(b_pad)) = mean(b(~isnan(b)));
    M      = padarray(ones(size(b)),[n/4,n/4]);
    M(n/4+1:3*n/4,n/4+1:3*n/4) = b~=0;
    if BW == 1
        figure(2), imagesc(b, [0,max(x_true(:))]), colormap(gray), colorbar, axis image
        yticks([9 29 49 69 89 109])
        yticklabels({'120', '100', '80', '60', '40', '20'})
        pause(.1);
    end

%% Section 3 - MCMC sampling %%
    %nsamps  = 10000;
    nsamps = 1000;
    l = zeros(n,n);
    l(1, 1) =  4; l(2 ,1) = -1;
    l(n,1) = -1; l(1 ,2) = -1;
    l(1,n) = -1; lhat = fft2(l);
    d = 2;
    accept_num = 0;
    accept_perc = zeros(nsamps-1,1);
    
    % Give values of nu and ell based on variogram fitting
    if BW == 1
        nu = 1;
        ellstart = 0.03;
        lambda = zeros(nsamps,1); lambda(1) = 10;
        delta = zeros(nsamps,1); delta(1) = 8.4958e-02;
    elseif BW == 2 && coloriter == 1
        nu = 1; 
        ellstart = 0.0364;
        lambda = zeros(nsamps,1); lambda(1) = 2.81e+03;
        delta = zeros(nsamps,1); delta(1) = 22;
    elseif BW == 2 && coloriter == 2
        nu = 1; 
        ellstart = 0.0313;
        lambda = zeros(nsamps,1); lambda(1) = 1.24e+03;
        delta = zeros(nsamps,1); delta(1) = 27; 
    elseif BW == 2 && coloriter == 3
        nu = 1; 
        ellstart = 0.0543;
        lambda = zeros(nsamps,1); lambda(1) = 1.51e+03;
        delta = zeros(nsamps,1); delta(1) = 22; 
    end
    alpha = zeros(nsamps,1); alpha(1) = delta(1)/lambda(1);
    ell = ellstart*ones(nsamps,1); ell(1) = ellstart;
    xsamp   = zeros(N,nsamps);
    
    % phat is parameterized differently to make it more numerically stable.
    % This means the delta chain is really delta*ell^(2*nu+d)/h^4 so, we
    % need to make the following modification at the end:
    % delta = delta./(ell.^(2*nu+d))*h^(2*nu+d)
    phat = real((lhat + (h/ell(1))^2).^(nu + d/2));
    
    a_params.ahat          = ahat;
    a_params.phat          = phat;
    a_params.lambda        = lambda(1);
    a_params.delta         = delta(1);
    a_params.M             = M;
    params.a_params        = a_params;
    Bmult                  = 'Bmult_DataDrivenMCMC';
    params.max_cg_iter     = 800;
    params.cg_step_tol     = 1e-4;
    params.grad_tol        = 1e-4;
    params.cg_io_flag   = 0;
    params.cg_figure_no = [];
    params.precond         = 'Amult_circulant';
    params.precond_params  = 1./(lambda(1)*abs(ahat).^2+delta(1)*phat);
    
    % RHS = lambda*A'D'b + eta1 + eta2 where eta1 = sqrt(lambda)*A'*D'*N(0,I)
    % and eta2 = sqrt(delta)*P^(1/2)*N(0,I). This is done so RHS is
    % N(A'D'b, (lambda*A'*D'*D*A+delta*P)) and so we have 
    % ((lambda*A'*D'*D*A+delta*P)x = N(A'D'b, (lambda*A'*D'*D*A+delta*P))
    % then x = N(inv(lambda*A'*D'*D*A+delta*P)lambda*A'D'b, inv(lambda*A'*D'*D*A+delta*P))
    % as desired. See pg 91 of John Bardsley's Inverse Problems book.
    AtDb                = lambda(1)*Amult_circulant(b_pad,conj(ahat));
    eta1                = sqrt(lambda(1))*real(ifft2(ahat.* ...
                            fft2(M.*randn(n,n))));
    eta2                = sqrt(delta(1))*real(ifft2(conj(phat).^(1/2).* ...
                            fft2(randn(n,n))));
    RHS                 = AtDb+eta1+eta2;
    xtemp               = CG(zeros(n,n),RHS,params,Bmult);
    nFFT                = 2;
    xsamp(:,1) = xtemp(:);

    % hyperpriors: lambda~Gamma(a,1/t0), delta~Gamma(a1,1/t1)
    a0=1; t0=0.0001; a1=1; t1=0.0001;
    hwait = waitbar(0,'MCMC samples in progress');
    tic
    for i=1:nsamps-1
        hwait = waitbar(i/nsamps);
        %------------------------------------------------------------------
        % 1a. Using conjugacy, sample the noise precision lam=1/sigma^2,
        % conjugate prior: lam~Gamma(a0,1/t0), mean = a0/t0, var = a0/t0^2.
        MAxtemp       = M.*real(ifft2(ahat.*fft2(xtemp))); % M*A*x
        MAxtemp       = MAxtemp(n/4+1:3*n/4,n/4+1:3*n/4);
        lambda(i+1) = gamrnd(a0+sum(sum(M==1))/2,1./(t0+norm(MAxtemp(:)-b(:))^2/2)); 
        %------------------------------------------------------------------
        % 1b. Using conjugacy, sample regularization precisions delta,
        % conjugate prior: delta~Gamma(a1,1/t1);
        Pxtemp = real(ifft2(phat.*fft2(xtemp)));
        delta(i+1) = gamrnd(a1+N/2,1./(t1+xtemp(:)'*Pxtemp(:)/2));
        %------------------------------------------------------------------
        
        if MCMCparam == 1
            lx = ell(i); % Equivalent to $l_k$ in paper
            
            %
            % The following complete the Metropolis-Hastings algorithm:
            %
            
            % Proposed sample
            ly = lognrnd(log(lx),.005); % Equivalent to $l^*$ in paper
            
            % Evaluate p(l_k|...)
            phat_x = real((lhat + (h/lx)^2).^(nu + d/2));
            Pxtemp = real(ifft2(phat_x.*fft2(xtemp)));
            pi_x = 1/2*sum(sum(log(phat_x)))-1/2*delta(i+1)*xtemp(:)'*Pxtemp(:);
            
            % Evaluate p(l^*|...)
            phat_y = real((lhat + (h/ly)^2).^(nu + d/2));
            Pxtemp = real(ifft2(phat_y.*fft2(xtemp)));
            pi_y = 1/2*sum(sum(log(phat_y)))-1/2*delta(i+1)*xtemp(:)'*Pxtemp(:);
            
            % Accept sample with probability beta(l^*,l_k)
            u = rand;
            if log(u) < pi_y - pi_x - log(lx) + log(ly) && ly>0
                ell(i+1) = ly;
                accept_num = accept_num + 1;
                accept_perc(i) = accept_num/i;
            else
                ell(i+1) = lx;
            end
           
            phat = real((lhat + h^2/ell(i+1)^2).^(nu + d/2));
            a_params.phat          = phat;
        end

        % 3. Using conjugacy relationships, sample the image using CG
        a_params.ahat          = ahat;
        a_params.lambda        = lambda(i+1);
        a_params.delta         = delta(i+1);
        a_params.M             = M;
        params.a_params        = a_params;
        Bmult                  = 'Bmult_DataDrivenMCMC';
        params.max_cg_iter     = 800;
        params.cg_step_tol     = 1e-4;
        params.grad_tol        = 1e-4;
        params.cg_io_flag      = 0;
        params.cg_figure_no    = [];
        params.precond         = 'Amult_circulant';
        params.precond_params  = 1./(lambda(i+1)*abs(ahat).^2+delta(i+1)*phat);
        AtDb            = lambda(i+1)*Amult_circulant(b_pad,conj(ahat));
        eta1            = sqrt(lambda(i+1))*real(ifft2(ahat ...
                            .*fft2(M.*randn(n,n))));
        eta2            = sqrt(delta(i+1))*real(ifft2(conj(phat).^(1/2) ...
                            .*fft2(randn(n,n))));
        RHS             = AtDb + eta1 + eta2;
        xtemp           = CG(zeros(n,n),RHS,params,Bmult);
        nFFT            = nFFT + 2;
        xsamp(:,i+1)    = xtemp(:);
    end
    toc
    close(hwait)     
    delta = delta./(ell.^(2*nu+d))*h^(2*nu+d); % Make modification
    alpha = delta./lambda;
    
%% Secion 4 - MCMC Results %%
% Remove the burn-in samples and visualize the MCMC chain
% Obtain the mean and 95% credibility intervals for x
    nburnin    = floor(nsamps/10); 
    xsamp      = xsamp(:,nburnin+1:end);
    q          = plims(xsamp(:,:)',[0.025,0.975]);
    x_mean     = mean(xsamp(:,:)')';
    clear xsamp % Save memory by clearing all samples

    % Extract middle 128x128 section of x_mean
    ind = (n/n2-1)/2;
    xmean = reshape(x_mean,n,n);
    xmean = xmean(ind*n2+1:(ind+1)*n2,ind*n2+1:(ind+1)*n2);
    q = reshape(q,2,n,n);
    q = q(1:2,ind*n2+1:(ind+1)*n2,ind*n2+1:(ind+1)*n2);
    if BW == 2
        xmeancolor(:,:,coloriter) = xmean;
        bcolor(:,:,coloriter) = b;
        Q(:,:,coloriter) = q(2,:,:)-q(1,:,:);
        x_truecolor(:,:,coloriter) = x_true;
        Lambda{coloriter} = lambda;
        Delta{coloriter} = delta;
        if MCMCparam == 4
            Ell{coloriter} = ell;
        end
    end
    relative_error = norm(x_true(:)-xmean(:))/norm(x_true(:));
 
%% Secion 5 - Plots %%
    figure(3)
      imagesc(xmean, [0,max(x_true(:))]), colorbar, colormap(gray), axis image

    figure(4), colormap(gray)
      imagesc(reshape(q(2,:,:)-q(1,:,:),n2,n2)), colorbar
      yticks([9 29 49 69 89 109])
      yticklabels({'120', '100', '80', '60', '40', '20'})
    % Output for individual chains, pairwise plots, and autocorrelation
    % for a subset of parameters.
    chains = [lambda(nburnin+1:end), delta(nburnin+1:end), alpha(nburnin+1:end)]';
    if MCMCparam == 1
        chains = [chains', ell(nburnin+1:end)]';
        names        = string(["\lambda","\delta","\alpha","\ell"]);
    else
        names        = string(["\lambda","\delta","\alpha"]);
    end
    fignum       = 5;
    [taux,acfun] = sample_plot(chains,names,fignum);
    % Plot the autocorrelations functions together.
    [~,nacf] = size(acfun);
    figure(8)
    plot((1:nacf),acfun(1,:),'r',(1:nacf),acfun(2,:),'k--','LineWidth',2), hold on
    plot((1:nacf),acfun(3,:),'c-o'), hold on
    if MCMCparam == 1
        plot((1:nacf),acfun(4,:),'b-+','LineWidth',2)
    end
    hold off
    axis([1,nacf,0,1])
    if MCMCparam == 1
        title('ACFs for $\lambda$, $\delta$, $\alpha$, and $\ell$.',...
            'interpreter','latex')
        legend(['$\lambda$: $\tau_{\rm int}(\lambda)=$',num2str(taux(1))],...
            ['$\delta$: $\tau_{\rm int}(\delta)=$',num2str(taux(2))], ...
            ['$\alpha$: $\tau_{\rm int}(\alpha)=$',num2str(taux(3))],...
            ['$\ell$: $\tau_{\rm int}(\ell)=$',num2str(taux(4))],...
            'interpreter','latex')
        accept_rate = zeros(nsamps-nburnin-1,1);
        for samp_num = 1:length(accept_rate)
            accept_rate(samp_num) = sum(ell(2:nburnin+samp_num+1)...
                - ell(1:nburnin+samp_num) ~= 0)/(nburnin+samp_num);
        end
        figure(9)
        plot(accept_rate)
        xlabel('Sample Number','interpreter','latex')
        ylabel('Acceptance Rate','interpreter','latex')
    else
        title('ACFs for $\lambda$, $\delta$, and $\alpha$.')
        legend(['$\lambda$: $\tau_{\rm int}(\lambda)=$',num2str(taux(1))],...
            ['$\delta$: $\tau_{\rm int}(\delta)=$',num2str(taux(2))], ... 
            ['$\alpha$: $\tau_{\rm int}(\alpha)=$',num2str(taux(3))],...
            'interpreter','latex')
    end
end
if BW == 2
    figure(10)
    imagesc(x_truecolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
    
    figure(11)
    imagesc(bcolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
    
    figure(12)
    imagesc(xmeancolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
end