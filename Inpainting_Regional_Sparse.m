%  
%  2d image demasking (inpainting) with periodic, data-driven boundary
%   conditions in the regional case. This m-file implements Bayesian
%   inpainting with either an isotropic Whittle-Matern prior (with
%   parameters selected using variograms) or an anisotropic Whittle-Matern
%   prior with different priors corresponds to different regions as chosen
%   by the user.
%  The regularization parameter that maximizes correlation is used.
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
% and white vs color and read in the data.  Modify line 31 to change image.
BW = input('Enter 1 to black and white deblur or 2 for color deblur. ');
prevmask = 0; % 1 for loading in a previous mask, 0 for drawing a new mask.
mask_type = 1; % 1 for random mask, 2 for systematic mask
maxiters = 5;
pmask = 0.6; % probability of a zero in b.

%%% Rock Image
x_read = imread('rock_wave_flickr2.jpg','jpg');
x_read = im2double(x_read);
[~,n,~] = size(x_read);
N = n^2;
sig = 0.004;

% Define initial vairables depending on whether we are dealing with black
% and white or color.
if BW == 1
    x_true = rgb2gray(x_read);
    x_true = x_true/max(x_true(:))*100;
    coloriters = 1;
elseif BW == 2
    xmapcolor=zeros(n/2,n/2,3);
    bcolor=zeros(n/2,n/2,3);
    x_truecolor=zeros(n/2,n/2,3);
    coloriters = 3;
end

%% Section 2 - Generate Masked Data %%
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
    % Create the mask by randomly zeroing out entries of b.
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
    Marray = false(n,n);
    Marray(n/4+1:3*n/4,n/4+1:3*n/4) = b~=0;
    % Create the large, sparse matrix that correpsonds to Marray
    M = spdiags(Marray(:),0,sparse(N,N)); % This matrix is n^2xn^2
    M1     = padarray(ones(size(b)),[n/4,n/4]);
    M1 = spdiags(M1(:),0,sparse(N,N));
    M(~any(M1,1),:)=[]; % This one is m^2xn^2 so that b = M*A*x_true+error
    

%% Section 3 - Initial Variogram %%
    d = 2; % dimension of the problem
    if coloriter == 1
        if BW == 1
            if prevmask == 1
                % Load in formerly created masks if required to do so.
                binaryImage = load('Regional_Masks');
                binaryImage = binaryImage.binaryImage;
                nregions = size(binaryImage,2);
            else
                % Display b and allow user to trace out the different regions
                [binaryImage, nregions] = region_select(b,[n n],1);
                pause(0.25);
                figure(2), imagesc(x_true, [0,max(x_true(:))])
                yticks([9 29 49 69 89 109])
                yticklabels({'120', '100', '80', '60', '40', '20'})
                colormap(gray), colorbar, axis image
                pause(0.25);
            end
        else
            if prevmask == 1
                % Load in formerly created masks if required to do so.
                binaryImage = load('Regional_Masks');
                binaryImage = binaryImage.binaryImage;
                nregions = size(binaryImage,2);
            else
                % Create and display b in collor and allow user to trace
                % out the different regions
                bcolor = x_read.*Marray;
                bcolor = bcolor(n/4+1:3*n/4,n/4+1:3*n/4,:);
                [binaryImage, nregions] = region_select(bcolor,[n n],1);
                pause(0.25);
            end
        end
    end

    vars = zeros(nregions,6);
    phat = cell(1,nregions);
    %Bmat = toeplitz([0; -1; sparse(n-2,1)],[0 1 sparse(1,n-2)]); % Zero BC
    Bmat = gallery('circul',[0 1 sparse(1,n-3) -1]); % Periodic BC
    L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n); % Both BCs
    L(n,1) = -1; L(1,n) = -1; % Periodic BC
    P = cell(1,nregions);
    for region = 1:nregions
        % Fit variograms for each region separately. We must limit the
        % inputs of the variogram to the non-zero elements of each region.
        binaryImageInterior = binaryImage{region}(n/4+1:3*n/4,n/4+1:3*n/4);
        bdf_region = bdf;
        bdf_region(:,3) = bdf(:,3).*binaryImageInterior(:);
        bdf_region(bdf_region(:,3)==0,:)=[];
        figure(6+region)
        [l1, l2, theta, nu] =...
            variogram_params_aniso(bdf_region(:,1:2), bdf_region(:,3));
        sgtitle(['$\theta$ = ', num2str(theta/pi*180), '$^\circ$, ratio = ',...
            num2str(l1/l2),', $\ell_1$ = ',num2str(l1),...
            ', $\ell_2$ = ',num2str(l2), ', $\nu$ = ', num2str(nu)],...
            'interpreter','latex','FontSize',14);
        pause(0.1);
        if l1/l2 >= 3 % Treat as isotropic if the ratio is less than 3.
            anis_flag = 1;
        else
            anis_flag = 0;
            S = variogram(bdf_region(:,1:2), bdf_region(:,3),...
                'nrbins',25,'maxdist',sqrt(2)/10,'subsample', Inf);
            nu = var_opt_nu_disc(2,S,10*h,max(S.val),min(S.val));
            [ell,psill,nugget,s]=variogramfit(S.distance,S.val,10*h,...
                max(S.val),S.num,'plotit',false,'model','matern',...
                'nu',nu,'weightfun','cressie85','nugget',0);
            l2 = 0;
        end
        % Store all variogram variables needed to construct the precisions
        if l2 == 0
            vars(region,:) = [l1, l2, theta, nu, 0, anis_flag];
        else
            vars(region,:) = [l1, l2, theta, nu, l1/l2, anis_flag];
        end
        

%% Section 4 - Prior and Initial Regularization Parameter, alpha %%
% Construct the precision matrices. The eigenvalues are stored in a cell
% phat and the full (sparse) matrix is stored in a cell P.
        if vars(region,6) == 1
            l1 = vars(region,1);
            l2 = vars(region,2);
            theta = vars(region,3);
            nu = vars(region,4);
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
            phat{region} = real((lhat + 1).^(nu + d/2));
            P{region} = (speye(n^2)+(at^2+bt^2)/h^2*kron(L,speye(n))...
                +(ct^2+dt^2)/h^2*kron(speye(n),L)...
                -2*(at*ct-bt*dt)/(4*h^2)*kron(Bmat,Bmat))^(nu+d/2);
        else
            ell = vars(region,1);
            nu = vars(region,4);
            l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
            l(n,1) = -1; l(1,n) = -1;
            lhat = fft2(ell^2/h^2*l);
            phat{region} = real((lhat + 1).^(nu + d/2));
            P{region} = (speye(n^2)+ell^2/h^2*(kron(L,speye(n))...
                +kron(speye(n),L)))^(nu+d/2);
        end
    end
    % Weight the precision matrices so the priors of each region have
    % approximately equal variance.
    w = prior_weights(phat,binaryImage);
    for region = 1:nregions
        phat{region} = phat{region}*w(region);
        P{region} = P{region}*w(region);
    end
    % Take a sample from the prior 
    figure(4)
    imagesc(prior_sample(binaryImage,phat)), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
    
    D = cell(1,nregions);
    for region = 1:nregions  
        % Replace diagonals with mask values
        D{region} = spdiags(binaryImage{region}(:),0,sparse(N,N));
    end

%% Section 5 - Iteratively Update nu, ell, and x_MAP %%
    dists = cell(1,nregions);
    nums = cell(1,nregions);
    for iteration=1:maxiters
        prevvars = vars;
        for region = 1:nregions
            if vars(region,6) == 1
                l1 = vars(region,1);
                l2 = vars(region,2);
                theta = vars(region,3);
                nu = vars(region,4);
                at = l2*sin(theta); % a_theta
                bt = l1*cos(theta); % b_theta
                ct = l2*cos(theta); % c_theta
                dt = l1*sin(theta); % d_theta
                lhat = fft2(1/h^2*(at^2+bt^2)*lx + 1/h^2*(ct^2+dt^2)*ly...
                   -2*(at*ct-bt*dt)/(4*h^2)*lxy);
                phat{region} = real((lhat + 1).^(nu + d/2));
                P{region} = (speye(n^2)+(at^2+bt^2)/h^2*kron(L,speye(n))...
                    +(ct^2+dt^2)/h^2*kron(speye(n),L)...
                    -2*(at*ct-bt*dt)/(4*h^2)*kron(Bmat,Bmat))^(nu+d/2);
            else
                ell = vars(region,1);
                nu = vars(region,4);
                l = zeros(n,n); l(1,1) = 4; l(1,2) = -1; l(2,1) = -1;
                l(n,1) = -1; l(1,n) = -1;
                lhat = fft2(N*ell^2*l);
                phat{region} = real((lhat + 1).^(nu + d/2));
                P{region} = (speye(n^2)+N*ell^2*(kron(L,speye(n))+kron(speye(n),L)))^(nu+d/2);
            end
        end
        w = prior_weights(phat,binaryImage);
        for region = 1:nregions
            phat{region} = phat{region}*w(region);
            P{region} = P{region}*w(region);
        end
        params.max_cg_iter  = 300;
        params.cg_step_tol  = 1e-4;
        params.grad_tol     = 1e-4;
        params.cg_io_flag   = 0;
        params.cg_figure_no = [];
        % Store information for the preconditioner
        params.precond      = 'Amult_circulant';
        % Store necessary info for matrix vector multiply (A*x) function
        % Assume P = D{1}*P{1}*D{1}+...+D{nr}*P{nr}*D{nr} to speed up algorithm 
        %   note that this gives an approximate alpha value. See notes in the
        %   Bmult_Regional.m file.
        Bmult               = 'Bmult_Regional';
        a_params.ahat       = ahat;
        a_params.phat       = phat;
        a_params.nregions   = nregions;
        a_params.Mask       = binaryImage;
        a_params.M          = Marray;
        params.a_params     = a_params;
        % Choose the regularization parameter
        AtMb                = Amult_circulant(b_pad,conj(ahat));
        disp(' *** Computing initial regularization parameter *** ')
        RegParam_fn = @(alpha) Best_Alpha_Regional(10^alpha,AtMb,params,...
            Bmult,x_true);
        alpha = 10^fminbnd(RegParam_fn,-16,0);
        %alpha = max(max(abs(ahat)))/max(max(abs([P{:}]))); %approximate alpha

        % Use CG to solve (A'*M'*M*A+alpha*P)x=A'*M'*b to get initial
        % x_MAP estimate where
        % P = inv(D{1}*inv(P{1})*D{1}+...+D{nr}*inv(P{nr})*D{nr})
        
        % The following sets up what is needed to compute Px.
        R = cell(1,nregions);
        ind_nz = cell(1,nregions); %index of nonzero diagonals
        DP = cell(1,nregions);
        PermMat = cell(1,nregions);
        for region = 1:nregions
            Dnot = speye(N)-D{region};
            Dnot = Dnot & diag(~diag(D{region})); % Equitvalent to Dnot - D{region}
            ind_nz{region} = find(diag(Dnot*P{region}));
            Preg = P{region}(ind_nz{region},ind_nz{region});
            r_ind = symamd(Preg); % Minimum degree reordering to increase efficiency.
            PermMat{region} = speye(size(r_ind,2));
            PermMat{region} = PermMat{region}(r_ind,:);
            R{region} = chol(Preg(r_ind,r_ind)); % Reordering
            DP{region} = D{region}*P{region};
        end

        a_params.M             = Marray;
        a_params.alpha         = alpha;
        a_params.R             = R;
        a_params.DP            = DP;
        a_params.D             = D;
        a_params.PermMat       = PermMat;
        a_params.nregions      = nregions;
        a_params.ind_nz        = ind_nz;
        params.a_params        = a_params;
        params.max_cg_iter     = 400;
        params.cg_step_tol     = 1e-4;%*norm(AtDb(:));
        params.grad_tol        = 1e-4;
        params.precond         = 'Amult_circulant';
        % The Bmult_DataDrivenWM function incorporates the WM prior.
        Bmult                  = 'Bmult_Regional_Sparse';
        % The following averages the phat values
        params.precond_params  = 1./(abs(ahat).^2+alpha*mean(cat(3,phat{:}),3));
        disp(' *** Computing the regularized solution *** ')
        xalpha     = CG(zeros(n,n),AtMb,params,Bmult);
        % This x_alpha estimate is n x n, not the smaller m x m
        % Take the middle m x m values of xalpha to get x_MAP
        xmap = xalpha(n/4+1:3*n/4,n/4+1:3*n/4);

        % Take only the xmap values in the region of interest for the
        % updated variogram fit.
        for region = 1:nregions
            binaryImageInterior = binaryImage{region}(n/4+1:3*n/4,n/4+1:3*n/4);
            xmapdf = [xcoord ycoord xmap(:)];
            xmapdf(:,3) = xmapdf(:,3).*binaryImageInterior(:);
            xmapdf(xmapdf(:,3)==0,:)=[];
            figure(6+region)
            if iteration == 1
                [l1, l2, theta, nu, dists{region}, nums{region}] = ...
                    variogram_params_aniso(xmapdf(:,1:2),xmapdf(:,3));
                pause(.1);
            else
                [l1, l2, theta, nu] = ...
                    variogram_params_aniso_dists(xmapdf(:,1:2),...
                    xmapdf(:,3),dists{region},nums{region});
                pause(.1);
            end
            if l1/l2 >= 3
                anis_flag = 1;
            else
                anis_flag = 0;
            end
            vars(region,:) = [l1 l2 theta nu l1/l2 anis_flag];
            if anis_flag == 0
                [S, dists{region}] = variogram(xmapdf(:,1:2),xmapdf(:,3),...
                    'nrbins',25,'maxdist',sqrt(2)/10,'subsample',m^2);
                nums{region} = S.num;
                % Again, find optimal nu
                nu = var_opt_nu_disc(2,S,1/vars(region,1),max(S.val),min(S.val));
                figure(2);
                [ell,psill,nugget,s]=variogramfit(S.distance,S.val,1/vars(region,1),...
                    max(S.val),S.num,'plotit',false,'model','matern','nu',nu,...
                    'weightfun','cressie85','nugget',min(S.val));
                vars(region,:) = [ell 0 theta nu 0 anis_flag];
            end
        end
        disp('Iteration, nu, l1, l2, theta, alpha')
        [iteration*ones(nregions,1) vars], disp(alpha)
        if sum(abs(prevvars(:,1)-vars(:,1)))/sum(prevvars(:,1)) < 0.01*nregions ...
           && sum(abs(prevvars(:,2)-vars(:,2)))/sum(prevvars(:,2)) < 0.01*nregions ...
           && sum(abs(prevvars(:,3)-vars(:,3))) == 0 ...
           && sum(abs(prevvars(:,4)-vars(:,4))) == 0
            clear dists
            break
        end
    end

%% Section 6 - Update Regularization Parameter using GCV.
    % Store CG iteration information for use within the regularization
    % parameter selection method.
    % Now we incorportate the prior information
    params.max_cg_iter  = 500;
    params.cg_step_tol  = 1e-3;
    params.grad_tol     = 1e-3;
    params.cg_io_flag   = 0;
    params.cg_figure_no = [];
    % Store information for the preconditioner
    params.precond      = 'Amult_circulant';
    % Store necessary info for matrix vector multiply (A*x) function
    a_params.ahat       = ahat;
    a_params.M          = Marray;
    a_params.phat       = phat;
    a_params.nregions   = nregions;
    a_params.Mask       = binaryImage;
    % Assume P = D{1}*P{1}*D{1}+...+D{nr}*P{nr}*D{nr} to speed up algorithm 
    %   note that this gives an approximate alpha value. See notes in the
    %   Bmult_Regional.m file.
    Bmult               = 'Bmult_Regional';
    params.a_params     = a_params;
    % Choose the regularization parameter
    AtMb                = Amult_circulant(b_pad,conj(ahat));
    disp(' *** Computing final regularization parameter *** ')
    RegParam_fn = @(alpha) Best_Alpha_Regional(10^alpha,AtMb,params,...
        Bmult,x_true);
    alpha = 10^fminbnd(RegParam_fn,-16,0);
%     RegParam_fn = @(alpha) GCV_DataDriven(10^alpha,b,AtDb,params,Bmult);
%     alpha = 10^fminbnd(RegParam_fn,-16,0) %

%% Section 7 - Final x_MAP Solution
    % With alpha in hand, tighten down on CG tolerances and use CG again
    % to solve (A'*M'*M*A+alpha*P)x=A'*M'*b. Then plot the results.
    R = cell(1,nregions);
    ind_nz = cell(1,nregions); %index of nonzero diagonals
    DP = cell(1,nregions);
    PermMat = cell(1,nregions);
    for region = 1:nregions
        Dnot = speye(N)-D{region};
        Dnot = Dnot & diag(~diag(D{region})); % Equitvalent to Dnot - D{region}
        ind_nz{region} = find(diag(Dnot*P{region}));
        Preg = P{region}(ind_nz{region},ind_nz{region});
        r_ind = symamd(Preg);
        PermMat{region} = speye(size(r_ind,2));
        PermMat{region} = PermMat{region}(r_ind,:);
        R{region} = chol(Preg(r_ind,r_ind)); % Reordering
        DP{region} = D{region}*P{region};
    end
    
    a_params.M             = Marray;
    a_params.alpha         = alpha;
    a_params.R             = R;
    a_params.DP            = DP;
    a_params.D             = D;
    a_params.PermMat       = PermMat;
    a_params.nregions      = nregions;
    a_params.ind_nz        = ind_nz;
    params.a_params        = a_params;
    params.max_cg_iter     = 1000;
    params.cg_step_tol     = 1e-4;
    params.grad_tol        = 1e-4;
    params.precond         = 'Amult_circulant';
    % The Bmult_DataDrivenWM function incorporates the sparse regional WM prior.
    Bmult                  = 'Bmult_Regional_Sparse';
    % The following averages the phat values
    params.precond_params  = 1./(abs(ahat).^2+alpha*mean(cat(3,phat{:}),3)); % 1 
    disp(' *** Computing final regularized solution *** ')
    % Use CG_Extended instead of CG to only compare inner region for
    %  convergence checks. This is used to slightly increase efficiency.
    [xalpha,iter_hist]     = CG_Extended(zeros(n,n),AtMb,params,Bmult);
    xmap = xalpha(n/4+1:3*n/4,n/4+1:3*n/4);
    % If doing color deblurring, store current images as one part of a 3-D array.
    if BW == 2
        xmapcolor(:,:,coloriter) = xmap;
        bcolor(:,:,coloriter) = b;
        x_truecolor(:,:,coloriter) = x_true;
    end
end

%% Secion 8 - Plots %%
if BW == 1
    figure(3)
    imagesc(xmap,[0,max(x_true(:))]), colorbar, colormap(gray), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
    
    figure(6),
    semilogy(iter_hist(:,2),'--k','LineWidth',2)
    title('Residual norm vs. CG iteration','FontSize', 14)
    norm(x_true-xmap,'fro')/norm(x_true,'fro')
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
    imagesc(xmapcolor), axis image
    yticks([9 29 49 69 89 109])
    yticklabels({'120', '100', '80', '60', '40', '20'})
end