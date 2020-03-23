function[weights] = prior_weights(phat,masks)
% This function assigns weights to the different regions of the prior for
% use in solving regional inverse problems, as are discussed in Chapter 5
% in the dissertation. Periodic boundary conditions are assumed. 
% The input are phat, the eigenvalues of the precision matrices for all the
% regions, and masks, the masking arrays to constrain the image to specific
% regions.
% Phat and masks should be of the same type. Either cells or doubles.
if ~isa(phat,'double') && ~isa(phat,'cell') || ~isa(masks,'double') && ~isa(masks,'cell')
    error('phat and masks must both be of type double or cell')
end
if isa(phat,'double')
    phat_double = phat;
    nregions = size(phat,3);
    phat = cell(1,1);
    for region = 1:nregions
        phat{region} = phat_double(:,:,region);
    end
    clear phat_double
end
if isa(masks,'double')
    masks_double = masks;
    nregions = size(masks,3);
    masks = cell(1,1);
    for region = 1:nregions
        masks{region} = masks_double(:,:,region);
    end
    clear masks_double
end
n = size(phat{1},1);
nregions = size(masks,2);
i = complex(0,1);
numsamps = 15; % Increase this number to add stability to the weights.
weights_samp = zeros(nregions,numsamps);
for j = 1:numsamps
    phat_samp = phat;
    repnum = 1;
    weights = zeros(nregions,1);
    E1 = randn(n,n);
    E2 = randn(n,n);
    % Loop through this until the weights are about the same to ensure the
    % variances are approxiametely equal.
    while max(abs(diff(weights(:,repnum)))) > 0.05 || repnum == 1
        % Here we take a sample from the prior.
        X = zeros(n,n);
        for region = 1:nregions
            X = X + real(masks{region}.*ifft2(phat_samp{region}.^(-1/2)...
                .*fft2(masks{region}.*(E1+i*E2))));
        end
        Xregion = cell(1,1);
        varregion = zeros(nregions,1);
        % Obtain the variance in each region
        for region = 1:nregions
            Xregion{region} = X.*masks{region};
            Xregion{region} = Xregion{region}(n/4+1:3*n/4,n/4+1:3*n/4);
            varregion(region) = var(Xregion{region}(Xregion{region}~=0));
        end
        % Assign weights proportional to the variances and ensure they add to 1. 
        w = 1/sum(varregion)*varregion;
        for region = 1:nregions
           phat_samp{region} = phat_samp{region}*w(region);
        end
        repnum = repnum + 1;
        weights(:,repnum) = w;
    end
    weights = weights(:,2:end);
    weight_prod = prod(weights,2);
    weights_samp(:,j) = weight_prod/sum(weight_prod);
end
weights = mean(weights_samp')';