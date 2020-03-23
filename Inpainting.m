%  
%  2d image demasking (inpainting) with periodic, data-driven boundary
%  conditions in the anisotropic case. This m-file implements Bayesian
%  inpainting with either an isotropic Whittle-Matern prior (with
%  parameters selected using variograms), an anisotropic Whittle-Matern
%  prior, a Tikhonov prior, or a Laplacian. The regularization parameter
%  that maximizes correlation is used.
%
%  Written by Rick Brown in 2019.
%
clear all
close all
path(path,'Functions&MatFiles')
path(path,'Variogram_Functions')
path(path,'Example_Images')

%% Section 1 - Color, Prior, and Reading Data %%
% Make selections about saving plots, whether to do the deblurring in black
% and white vs color, what prior to use, and then read in the data.
% Comment out lines 29-34 and uncomment lines 37-42 to use Main Hall image
% instead of the Rock image. 
BW = input('Enter 1 to black and white deblur or 2 for color deblur. ');
prior = input(['Enter 1 to use an isotropic prior, 2 to use a geometric anisotropic prior,\n'...
    '       3 to use a Tikhonov (identity) prior, or 4 to use a Laplacian prior. ']);
mask_type = 1; % 1 for random mask, 2 for systematic mask (grid)
maxiters = 10; % Maximum number of iterations for the semivariogram method.

%%% Rock Image
% x_read = imread('rock_wave_flickr.jpg','jpg');
% x_read = im2double(x_read);
% [~,n,~] = size(x_read);
% sig = 0.004;
% pmask = 0.6; % probability of a zero in b.
% im = 'wave';

% %%% Main hall
x_read = imread('Main_Hall_2008.jpg','jpg');
x_read = im2double(x_read);
[~,n,~] = size(x_read);
sig = 0.002;
pmask = 0.4; % probability of a zero in b.
im = 'hall';

% Define initial vairables depending on whether we are dealing with black
% and white or color.
if BW == 1
    x_true = rgb2gray(x_read);
    x_true = x_true/max(x_true(:))*100;
    coloriters = 1;
    if prior == 1
    	var_hist = zeros(coloriters, 3);
    elseif prior == 2
    	var_hist = zeros(coloriters, 5);
    else
        alpha_hist = zeros(coloriters,1);
    end
elseif BW == 2
    xmapcolor=zeros(n/2,n/2,3);
    bcolor=zeros(n/2,n/2,3);
    x_truecolor=zeros(n/2,n/2,3);
    coloriters = 3;
    if prior == 1
    	var_hist = zeros(coloriters, 3);
    elseif prior == 2
    	var_hist = zeros(coloriters, 5);
    else
        alpha_hist = zeros(coloriters,1);
    end
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
    if strcmp(im,'hall')
        x      = (-0.5+h/2:h:0.5-h/2)';
        [X,Y]  = meshgrid(x);
        % The kernel adds a slight burr to the image
        kernel = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
    elseif strcmp(im,'wave')
        % The kernel used here corresponds to the identity matrix.
        kernel = zeros(n,n);
        kernel(n/2+1,n/2+1) = 1;
    end
    ahat   = fft2(fftshift(kernel));
    Ax     = real(ifft2(ahat.*fft2(x_true)));

    % Extract 128x128 middle subregion from Ax and x, add noise and plot.
    Ax     = Ax(n/4+1:3*n/4,n/4+1:3*n/4);
    x_true = x_true(n/4+1:3*n/4,n/4+1:3*n/4);

    [~,m]= size(Ax);
    err_lev= 5;
    noise  = err_lev/100 * norm(Ax(:)) / sqrt(m^2);
    rng(0) % fixes the seed of the random number generator.
    eta = noise*randn(m,m);
    b      = Ax + eta;
    if BW == 1
        figure(1), imagesc(x_true, [0,max(x_true(:))]), colormap(gray), colorbar, axis image
        yticks([9 29 49 69 89 109])
        yticklabels({'120', '100', '80', '60', '40', '20'})
        pause(.1);
    end
    xcoord = repelem((1:m)',m)/m; % Assign an x coordinate to each b element
    ycoord = repmat((m:-1:1)',m,1)/m; % Assign an y coordinate to each b element
    bdf = [xcoord ycoord b(:)];
    if mask_type == 1
        rng(0)
        bdf = [xcoord ycoord binornd(1,1-pmask,m^2,1).*b(:)];
        b = reshape(bdf(:,3),m,m);
    elseif mask_type == 2
        Amask = eye(m);
        for i=1:m/4-1
            Amask(4*i:4*i+1,4*i:4*i+1)=zeros(2,2);
        end
        b = Amask*b*Amask;
        bdf = [xcoord ycoord b(:)];
    end
    % zero pad b and create the mask that matches the zeroed-out values in b.
    b_pad = padarray(b,[n/4,n/4]);
    b_pad(isnan(b_pad)) = mean(b(~isnan(b)));
    bp_hat = fft2(b_pad);
    M      = padarray(ones(size(b)),[n/4,n/4]);
    M(n/4+1:3*n/4,n/4+1:3*n/4) = b~=0;
    if BW == 1
        figure(2), imagesc(b, [0,max(x_true(:))]), colormap(gray), colorbar, axis image
        yticks([9 29 49 69 89 109])
        yticklabels({'120', '100', '80', '60', '40', '20'})
        pause(.1);
    end

%% Section 3 - Initial Variogram %%
    if prior == 1 || prior == 2
        d = 2; % dimension of the problem
        % Remove all values and the coordinates of those values where b=0.
        bdf(bdf(:,3)==0,:)=[];
        bvec = bdf(:,3);
        if prior == 2
            figure(7)
            [l1, l2, theta, nu] =...
                variogram_params_aniso(bdf(:,1:2), bvec);
            sgtitle(['$\theta$ = ', num2str(theta/pi*180), '$^\circ$, ratio = ',...
            num2str(l1/l2),'. $\ell_1$ = ',num2str(l1),...
            ', $\ell_2$ = ',num2str(l2)],'interpreter','latex','FontSize',14);
            pause(0.1);
        elseif prior == 1
            S = variogram(bdf(:,1:2), bdf(:,3),'nrbins',25,...
                'maxdist',sqrt(2)/10,'subsample', Inf);
            nu = var_opt_nu_disc(2,S,10*h,max(S.val),min(S.val));
             % Create plot of best variogram fit with optimal nu and ell
            [ell,psill,nugget,s]=variogramfit(S.distance,S.val,10*h,...
                max(S.val),S.num,'plotit',false,'model','matern',...
                'nu',nu,'weightfun','cressie85','nugget',0);
        end
    
        
%% Section 4 - Prior and Initial Regularization Parameter, alpha %%
        % Now obtain L and P for periodic BC
        if prior == 2
            at = l2*sin(theta); % a_theta
            bt = l1*cos(theta); % b_theta
            ct = l2*cos(theta); % c_theta
            dt = l1*sin(theta); % d_theta
            lx = zeros(n,n); lx(1,1) = 2; lx(1,2) = -1; lx(1,n) = -1;
            ly = zeros(n,n); ly(1,1) = 2; ly(2,1) = -1; ly(n,1) = -1;
            lxy = zeros(n,n); lxy(2,2) = 1; lxy(2,n) = -1; lxy(n,2) = -1; lxy(n,n) = 1;
            lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
               -2*(at*ct-bt*dt)/(4*h^2)*lxy);
            phat = real((lhat + 1).^(nu + d/2));
        elseif prior == 1
            l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
            l(n,1) = -1; l(1,n) = -1;
            lhat = fft2(l);
            phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
        end


%% Section 5 - Iteratively Update nu, ell, and x_MAP %%
        for iteration=1:maxiters
            if prior == 2 
                prevl1 = l1;
                prevl2 = l2;
                prevtheta = theta;
                prevnu = nu;
                at = l2*sin(theta); % a_theta
                bt = l1*cos(theta); % b_theta
                ct = l2*cos(theta); % c_theta
                dt = l1*sin(theta); % d_theta
                lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
                   -2*(at*ct-bt*dt)/(4*h^2)*lxy);
                phat = real((lhat + 1).^(nu + d/2));
            elseif prior == 1
                prevell = ell;
                prevnu = nu;
                phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
                a_params.phat       = phat;
            end
            params.max_cg_iter  = 300;
            params.cg_step_tol  = 1e-4;
            params.grad_tol     = 1e-4;
            params.cg_io_flag   = 0;
            params.cg_figure_no = [];
            % Store information for the preconditioner
            params.precond      = 'Amult_circulant';
            % Store necessary info for matrix vector multiply (A*x) function
            Bmult               = 'Bmult_DataDrivenWM';
            a_params.ahat       = ahat;
            a_params.phat       = phat;
            a_params.M          = M;
            params.a_params     = a_params;
            % Choose the regularization parameter
            AtMb                = Amult_circulant(b_pad,conj(ahat));
            disp(' *** Computing initial regularization parameter *** ')
            RegParam_fn = @(alpha) Best_Alpha(10^alpha,AtMb,params,Bmult,x_true);
            alpha = 10^fminbnd(RegParam_fn,-16,0);
            
            % Use CG again to solve (A'*M'*M*A+alpha*P)x=A'*M'*b to get initial
            % x_MAP estimate where P = (I + (ell/h)^2*L)^(nu+d/2)
            a_params.alpha         = alpha;
            a_params.phat          = phat;
            params.a_params        = a_params;
            Bmult                  = 'Bmult_DataDrivenWM';
            params.max_cg_iter     = 800;
            params.cg_step_tol     = 1e-5;
            params.grad_tol        = 1e-5;
            params.precond         = 'Amult_circulant';
            params.precond_params  = 1./(abs(ahat).^2+alpha*phat);
            disp(' *** Computing the regularized solution *** ')
            xalpha     = CG(zeros(n,n),AtMb,params,Bmult);
            % This x_alpha estiamte is n x n, not the smaller m x m
            % Take the middle m x m values of xalpha to get x_MAP
            xmap = xalpha(n/4+1:3*n/4,n/4+1:3*n/4);
            if prior == 2
                if iteration == 1
                    [l1, l2, theta, nu, dists, nums] = ...
                        variogram_params_aniso([xcoord ycoord],xmap(:));
                else
                    [l1, l2, theta, nu] = ...
                        variogram_params_aniso_dists([xcoord ycoord],...
                        xmap(:),dists,nums);
                end
                if abs(prevl1-l1)/prevl1<.01 && prevnu == nu...
                        && abs(prevl2-l2)/prevl2<0.01 ...
                        && prevtheta == theta
                    break
                end
                disp('Iteration, nu, ell_1, ell_2, theta, alpha')
                [iteration nu l1 l2 theta alpha] % Output some info each iteration
            elseif prior == 1
                if iteration == 1
                    [S, dists] = variogram([xcoord ycoord],xmap(:),...
                        'nrbins',25,'maxdist',sqrt(2)/10,'subsample',m^2);
                else
                    S=variogram_dists([xcoord ycoord],xmap(:),dists,S.num,...
                        'nrbins',25,'maxdist',sqrt(2)/10,'subsample',m^2);
                end
                % Again, find optimal nu
                nu = var_opt_nu_disc(2,S,ell,psill,nugget);
                [ell,psill,nugget,s]=variogramfit(S.distance,S.val,ell,...
                    psill,S.num,'plotit',false,'model','matern','nu',nu,...
                    'weightfun','cressie85','nugget',nugget);
                if abs(prevell-ell)/prevell<.01 && prevnu == nu
                    break
                end
                disp('Iteration, nu, ell, alpha')
                [iteration nu ell alpha] % Output some info each iteration
            end
        end
    end

%% Section 6 - Update Regularization Parameter.
    % Store CG iteration information for use within the regularization
    % parameter selection method.
    % Now we incorportate the prior information
    params.max_cg_iter  = 800;
    params.cg_step_tol  = 1e-6;
    params.grad_tol     = 1e-6;
    params.cg_io_flag   = 0;
    params.cg_figure_no = [];
    % Store information for the preconditioner
    params.precond      = 'Amult_circulant';
    % Store necessary info for matrix vector multiply (A*x) function
    a_params.ahat       = ahat;
    a_params.M          = M;
    if prior == 2
        at = l2*sin(theta); % a_theta
        bt = l1*cos(theta); % b_theta
        ct = l2*cos(theta); % c_theta
        dt = l1*sin(theta); % d_theta
        lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
           -2*(at*ct-bt*dt)/(4*h^2)*lxy);
        phat = real((lhat + 1).^(nu + d/2));
        Bmult               = 'Bmult_DataDrivenWM';
        a_params.phat       = phat;
    elseif prior == 1
        phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
        Bmult               = 'Bmult_DataDrivenWM';
        a_params.phat       = phat;
    elseif prior == 4
        l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
        l(n,1) = -1; l(1,n) = -1;
        lhat = fft2(l);
        phat = lhat.^2;
        Bmult               = 'Bmult_DataDrivenWM';
        a_params.phat       = phat;
    else
        Bmult               = 'Bmult_DataDriven';
        a_params.phat       = ones(n,n);
    end
    params.a_params     = a_params;
    % Choose the regularization parameter
    AtMb                = Amult_circulant(b_pad,conj(ahat));
    disp(' *** Computing final regularization parameter *** ')
    RegParam_fn = @(alpha) Best_Alpha(10^alpha,AtMb,params,Bmult,x_true);
    alpha = 10^fminbnd(RegParam_fn,-16,0);
%     RegParam_fn = @(alpha) GCV_DataDriven(alpha,b,AtDb,params,Bmult);
%     alpha = fminbnd(RegParam_fn,0,1)
    if prior == 1
    	var_hist(coloriter,:) = [nu ell alpha];
    elseif prior == 2
    	var_hist(coloriter,:) = [nu l1 l2 theta alpha];
    else
        alpha_hist(coloriter) = alpha;
    end

%% Section 7 - Final x_MAP Solution
    % With alpha in hand, tighten down on CG tolerances and use CG again
    % to solve (A'*M'*M*A+alpha*P)x=A'*M'*b. Then plot the results.
    a_params.alpha         = alpha;
    params.a_params        = a_params;
    params.max_cg_iter     = 3500;
    params.cg_step_tol     = 1e-6;
    params.grad_tol        = 1e-6;
    params.precond         = 'Amult_circulant';
    % The Bmult_DataDrivenWM function incorporates the WM prior.
    if prior ~= 3
        Bmult                  = 'Bmult_DataDrivenWM';
        params.precond_params  = 1./(abs(ahat).^2+alpha*phat);
    else
        Bmult                  = 'Bmult_DataDriven';
        params.precond_params  = 1./(abs(ahat).^2+alpha);
    end
    disp(' *** Computing final regularized solution *** ')
    [xalpha,iter_hist]     = CG(zeros(n,n),AtMb,params,Bmult);
    xmap = xalpha(n/4+1:3*n/4,n/4+1:3*n/4);
    
    % If doing color deblurring, store current images as one part of a 3-D array.
    if BW == 2
        xmapcolor(:,:,coloriter) = xmap;
        bcolor(:,:,coloriter) = b;
        x_truecolor(:,:,coloriter) = x_true;
    end
end
if BW == 1
    if prior == 1
        xmap_iso = xmap;
    elseif prior == 2
        xmap_anis = xmap;
    elseif prior == 3
        xmap_tik = xmap;
    end
else
    if prior == 1
        xmapcolor_iso = xmapcolor;
    elseif prior == 2
        xmapcolor_anis = xmapcolor;
    elseif prior == 3
        xmapcolor_tik = xmapcolor;
    elseif prior == 4
        xmapcolor_lap = xmapcolor;
    end
end

%% Secion 8 - Plots %%
if BW == 1
    figure(3)
    imagesc(xmap,[0,max(x_true(:))]), colorbar, colormap(gray), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    figure(6),
    semilogy(iter_hist(:,2),'k','LineWidth',2)
    title('Residual norm vs. CG iteration','FontSize', 14)
    
    relative_error = norm(x_true-xmap,'fro')/norm(x_true,'fro'); 
else
    figure(3)
    imagesc(x_truecolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    figure(4)
    % Multiply by 1.1 below to make color look more natural
    imagesc(bcolor*1.1), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    figure(5)
    imagesc(xmapcolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
end