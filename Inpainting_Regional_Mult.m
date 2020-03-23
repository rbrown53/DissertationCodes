%  
%  2d image demasking (inpainting) with periodic, data-driven boundary
%   conditions in the regional case. This m-file implements Bayesian
%   inpainting with either an isotropic Whittle-Matern prior (with
%   parameters selected using variograms) or an anisotropic Whittle-Matern
%   prior with different priors corresponds to different regions as chosen
%   by the user. Rather than using sparse matrices to solve one inverse
%   problem, this solves a different inverse problem for each region using
%   DFTs and stitches them all together.
%  The regularization parameter that maximizes correlation is used.
% 
%  This code was not used in the dissertation.
%
%  Written by Rick Brown in 2019.
%
clear all
close all
path(path,'Functions&MatFiles')
path(path,'Variogram_Functions')
path(path,'Deblur_Images')

%% Section 1 - Color, Prior, and Reading Data %%
% Make selections about saving plots, whether to do the deblurring in black
% and white vs color, whether to use the Whittle-Matern prior as opposed to
% the identity, and read in the data. Modify line 36 to change image. 
BW = input('Enter 1 to black and white deblur or 2 for color deblur. ');
prevmask = 0; % 1 for loading in a previous mask, 0 for a new mask.
mask_type = 1; % 1 for random mask, 2 for systematic mask
maxiters = 5;
pmask = 0.6; % probability of a zero in b.

%%% Rock Image
x_read = imread('rock_wave_flickr2.jpg','jpg');
x_read = im2double(x_read);
[~,n,~] = size(x_read);
sig = 0.004;

% Define initial vairables depending on whether we are dealing with black
% and white or color.
if BW == 1
    x_true = rgb2gray(x_read);
    x_true = x_true/max(x_true(:))*100;
    coloriters = 1;
elseif BW == 2
    xmapcolor_mix=zeros(n/2,n/2,3);
    bcolor=zeros(n/2,n/2,3);
    x_truecolor=zeros(n/2,n/2,3);
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
    x      = (-0.5+h/2:h:0.5-h/2)';
    [X,Y]  = meshgrid(x);
    %kernel = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
    kernel = zeros(n,n);
    kernel(n/2+1,n/2+1) = 1;
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
    M = false(n,n);
    M(n/4+1:3*n/4,n/4+1:3*n/4) = b~=0;

%% Section 3 - Initial Variogram %%
    d = 2; % dimension of the problem
    if coloriter == 1
        bcolor = x_read.*M;
        bcolor = bcolor(n/4+1:3*n/4,n/4+1:3*n/4,:);
        if BW == 1
            figure(1), imagesc(x_true, [0,max(x_true(:))]), colormap(gray), colorbar, axis image
            yticks([9 29 49 69 89 109])
            yticklabels({'120', '100', '80', '60', '40', '20'})
            pause(0.1);
            [binaryImage, nregions] = region_select(b,[n n],2);
            pause(0.1);
        else
            if prevmask == 1
                % Load in formerly created masks.
                binaryImage = load('Inpainting_Wave2_Mask');
                binaryImage = binaryImage.binaryImage;
            else
                [binaryImage, nregions] = region_select(bcolor,[n n],1);
                pause(0.1);
            end
        end
    end
    vars = zeros(nregions,6);
    xalpha_all = cell(1,nregions);
    xmap = cell(1,nregions);
    for i = 1:nregions
        % The following is equivalent to M.*binaryImage{i} but keeps Mask
        % of type logical.
        Mask = M & binaryImage{i};
        binaryImageInterior = binaryImage{i}(n/4+1:3*n/4,n/4+1:3*n/4);
        bdf_region = bdf;
        bdf_region(:,3) = bdf(:,3).*binaryImageInterior(:);
        bdf_region(bdf_region(:,3)==0,:)=[];
        figure(6+i)
        [l1, l2, theta, nu] =...
            variogram_params_aniso(bdf_region(:,1:2), bdf_region(:,3));
        sgtitle(['$\theta$ = ', num2str(theta/pi*180), '$^\circ$, ratio = ',...
            num2str(l1/l2),', $\ell_1$ = ',num2str(l1),...
            ', $\ell_2$ = ',num2str(l2)],'interpreter','latex','FontSize',14);
        pause(0.1);
        if l1/l2 >= 2.5
            anis_flag = 1;
        else
            anis_flag = 0;
            S = variogram(bdf_region(:,1:2), bdf_region(:,3),...
                'nrbins',25,'maxdist',sqrt(2)/10,'subsample', Inf);
            nu = var_opt_nu_disc(2,S,10*h,max(S.val),min(S.val));
            % Fit variogram
            [ell,psill,nugget,s]=variogramfit(S.distance,S.val,10*h,...
                max(S.val),S.num,'plotit',false,'model','matern',...
                'nu',nu,'weightfun','cressie85','nugget',0);
            l2 = 0;
        end
        if l2 == 0
            vars(i,:) = [l1, 0, theta, nu, 0, anis_flag];
        else
            vars(i,:) = [l1, l2, theta, nu, l1/l2, anis_flag];
        end

%% Section 4 - Prior and Initial Regularization Parameter, alpha %%
    % Now obtain L and P for periodic BC
        if vars(i,6) == 1
            l1 = vars(i,1);
            l2 = vars(i,2);
            theta = vars(i,3);
            nu = vars(i,4);
            at = l2*sin(theta); % a_theta
            bt = l1*cos(theta); % b_theta
            ct = l2*cos(theta); % c_theta
            dt = l1*sin(theta); % d_theta
            lx = zeros(n,n); lx(1,1) = 2; lx(1,2) = -1; lx(1,n) = -1;
            ly = zeros(n,n); ly(1,1) = 2; ly(2,1) = -1; ly(n,1) = -1;
            lxy = zeros(n,n); lxy(2,2) = 1;lxy(2,n) = -1;
            lxy(n,2) = -1; lxy(n,n) = 1;
            lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
               -2*(at*ct-bt*dt)/(4*h^2)*lxy);
            phat = real((lhat + 1).^(nu + d/2));
        else
            ell = vars(i,1);
            nu = vars(i,4);
            l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
            l(n,1) = -1; l(1,n) = -1;
            lhat = fft2(l);
            phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
        end

%% Section 5 - Iteratively Update nu, ell, and x_MAP %%
        for iteration=1:maxiters
            if vars(i,6) == 1
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
            else
                prevell = ell;
                prevnu = nu;
                phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
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
            a_params.M          = Mask;
            params.a_params     = a_params;
            % Choose the regularization parameter
            AtMb                = Amult_circulant(b_pad.*Mask,conj(ahat));
            disp(' *** Computing initial regularization parameter *** ')
            RegParam_fn = @(alpha) Best_Alpha_Mask(10^alpha,AtMb,params,...
                Bmult,x_true,binaryImageInterior);
            alpha = 10^fminbnd(RegParam_fn,-16,0);

            % Use CG again to solve (A'*A+alpha*P)x=A'*b to get initial
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
            % This x_alpha estimate is n x n, not the smaller m x m
            % Take the middle m x m values of xalpha to get x_MAP
            xmap{i} = xalpha(n/4+1:3*n/4,n/4+1:3*n/4);
            
            % Take only the xmap values in the region of interest for the
            % updated variogram fit.
            xmapdf = [xcoord ycoord xmap{i}(:)];
            xmapdf(:,3) = xmapdf(:,3).*binaryImageInterior(:);
            xmapdf(xmapdf(:,3)==0,:)=[];
            if vars(i,6) == 1
                if iteration == 1
                    [l1, l2, theta, nu, dists, nums] = ...
                        variogram_params_aniso(xmapdf(:,1:2),xmapdf(:,3));
                    pause(.1);
                else
                    [l1, l2, theta, nu] = ...
                        variogram_params_aniso_dists(xmapdf(:,1:2),...
                        xmapdf(:,3),dists,nums);
                    pause(.1);
                end
                if abs(prevl1-l1)/prevl1<.01 && prevnu == nu...
                        && abs(prevl2-l2)/prevl2<0.01 && prevtheta == theta
                    break
                end
                disp('Iteration, nu, l1, l2, theta, alpha')
                [iteration nu l1 l2 theta alpha]
            else
                if iteration == 1
                    [S, dists] = variogram(xmapdf(:,1:2),xmapdf(:,3),...
                        'nrbins',25,'maxdist',sqrt(2)/10,'subsample',m^2);
                else
                    S=variogram_dists(xmapdf(:,1:2),xmapdf(:,3),dists,S.num,...
                        'nrbins',25,'maxdist',sqrt(2)/10,'subsample',m^2);
                end
                % Again, find optimal nu
                nu = var_opt_nu_disc(2,S,ell,psill,nugget);
                figure(2);
                [ell,psill,nugget,s]=variogramfit(S.distance,S.val,ell,...
                    psill,S.num,'plotit',false,'model','matern','nu',nu,...
                    'weightfun','cressie85','nugget',nugget);
                if abs(prevell-ell)/prevell<.01 && prevnu == nu
                    break
                end
                disp('Iteration, nu, ell, alpha')
                [iteration nu ell alpha]
            end
        end

%% Section 6 - Update Regularization Parameter using GCV.
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
        a_params.M          = Mask;
        if vars(i,6) == 1
            at = l2*sin(theta); % a_theta
            bt = l1*cos(theta); % b_theta
            ct = l2*cos(theta); % c_theta
            dt = l1*sin(theta); % d_theta
            lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
               -2*(at*ct-bt*dt)/(4*h^2)*lxy);
            phat = real((lhat + 1).^(nu + d/2));
            Bmult               = 'Bmult_DataDrivenWM';
            a_params.phat       = phat;
        else
            phat = real(((ell/h)^2*lhat + 1).^(nu + d/2));
            Bmult               = 'Bmult_DataDrivenWM';
            a_params.phat       = phat;
        end
        params.a_params     = a_params;
        % Choose the regularization parameter
        AtMb                = Amult_circulant(b_pad.*Mask,conj(ahat));
        disp(' *** Computing final regularization parameter *** ')
        RegParam_fn = @(alpha) Best_Alpha_Mask(10^alpha,AtMb,params,...
            Bmult,x_true,binaryImageInterior);
        alpha = 10^fminbnd(RegParam_fn,-16,0);
    %     RegParam_fn = @(alpha) GCV_DataDriven(alpha,b,AtDb,params,Bmult);
    %     alpha = fminbnd(RegParam_fn,0,1)

    %% Section 7 - Final x_MAP Solution
        % With alpha in hand, tighten down on CG tolerances and use CG again
        % to solve (A'*A+alpha*P)x=A'*b. Then plot the results.
        a_params.alpha         = alpha;
        params.a_params        = a_params;
        params.max_cg_iter     = 3500;
        params.cg_step_tol     = 1e-6;
        params.grad_tol        = 1e-6;
        params.precond         = 'Amult_circulant';
        % The Bmult_DataDrivenWM function incorporates the WM prior.
        Bmult                  = 'Bmult_DataDrivenWM';
        params.precond_params  = 1./(abs(ahat).^2+alpha*phat);
        disp(' *** Computing final regularized solution *** ')
        [xalpha,iter_hist]     = CG(zeros(n,n),AtMb,params,Bmult);
        xalpha_all{i} = xalpha.*binaryImage{i};
        xmap{i} = xalpha_all{i}(n/4+1:3*n/4,n/4+1:3*n/4);
    end
    % Combine all different xmaps from the different regions. If any values
    % were in multiple regions, average them for the final solution.
    xmap_stitch = zeros(m,m);
    for i = 1:nregions
        for j=1:m
            for k=1:m
                if xmap_stitch(j,k) == 0 && xmap{i}(j,k) ~= 0 
                    xmap_stitch(j,k) = xmap{i}(j,k);
                elseif xmap_stitch(j,k) ~= 0 && xmap{i}(j,k) ~= 0
                    xmap_stitch(j,k) = (xmap_stitch(j,k)+xmap{i}(j,k))/2;
                end
            end
        end
    end
    % If doing color deblurring, store current images as one part of a 3-D array.
    if BW == 2
        xmapcolor_mix(:,:,coloriter) = xmap_stitch;
        bcolor(:,:,coloriter) = b;
        x_truecolor(:,:,coloriter) = x_true;
    end
end

%% Secion 8 - Plots %%
if BW == 1
    figure(3)
    imagesc(xmap_stitch,[0,max(x_true(:))]), colorbar, colormap(gray), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    norm(x_true-xmap_stitch,'fro')/norm(x_true,'fro')
else
    figure(3)
    imagesc(x_truecolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    figure(4)
    imagesc(bcolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})

    figure(5)
    imagesc(xmapcolor_mix), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
end