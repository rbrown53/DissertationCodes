function [l1, l2, theta, nu, dists_dir, nums] = variogram_params_aniso(xycoords, vec)
% This function computes directional semivariograms for a given field,
%   determines theta, the angle of maximum anisotropy, and the ratio of
%   anisotropy. Then transforms to a new coordinate system to fit an
%   omnidirectional semivariogram to obtain estimates for nu and l1. l2 is
%   then computed using l2 = l1/ratio.
% 
% The distances between all coordinates (dists_dir) and the number of
%   values at those distances (nums) are also returned to increase
%   efficiently in the next iteration. The function
%   variogram_params_aniso_dists should be used after the first iteration.
% 
% The inputs are:
% xycoords = spatial coordinates of the field to which we want to fit 
%   directional semivariograms
% vec = values of the field at each spatial coordinate

% Written by Rick Brown

% Compute initial directional variograms with spacing of 15 degrees
disp(' *** Computing directional variograms *** ')
maxdist = sqrt(sum((max(xycoords)-min(xycoords)).^2));
[S_dir, dists_dir] = variogram(xycoords,vec,'nrbins',25,'maxdist',maxdist/2,...
    'subsample', Inf,'anisotropy',true,'thetastep',15);
ntheta = length(S_dir.theta)-1;
nums = S_dir.num;
% Convert angles from those with respect to the y-axis to those with
% respect to the x-axis
xtheta = zeros(ntheta,1);
for i = 1:ntheta
    if S_dir.theta(i) <= pi/2
        xtheta(i) = acos(sin(S_dir.theta(i)));
    else
        xtheta(i) = -acos(sin(S_dir.theta(i)));
    end
end
%% Find the distance at which the smoothed variogram value hits the critical gamma value.
% multiple gamma crit values are used and the median distance is used for
% the range.
ranges = zeros(ntheta,1);
ratio = zeros(10,1);
critgam = zeros(10,1);
theta = 10*ones(10,1);
for j = 1:10
    critgam(j) = min(max(S_dir.val)) + 0.06*(2-j)*min(max(S_dir.val));
    if critgam(j) >= min(min(S_dir.val))  
        for i = 1:ntheta
            smoothSval = smoothdata(S_dir.val(:,i),'loess',20); 
            range_val = smoothSval >= critgam(j);
            range_ind = find(range_val==1,1,'first');
            if isempty(range_ind) == 1
                ranges(i) = 0;
            elseif range_ind == 1
                ranges(i) = S_dir.distance(range_ind);
            else
                ranges(i) = (S_dir.distance(range_ind)+S_dir.distance(range_ind-1))/2;
            end
        end
        delta = zeros(6,1);
        for i = 1:6
            delta(i) = max(ranges(i)/ranges(i+6), ranges(i+6)/ranges(i));
        end
        if ~isempty(max(delta(~isinf(delta) & delta~=0)))
            [ratio(j),max_index] = max(delta(~isinf(delta) & delta~=0));
            range1 = ranges(max_index);
            range2 = ranges(max_index+6);
            if range1 > range2
                theta(j) = xtheta(max_index);
            else
                theta(j) = xtheta(max_index+6);
            end
        else
            ratio(j) = NaN;
            theta(j) = NaN;
        end
    end
end
theta_pos = theta == mode(theta(theta~=10 & ~isnan(theta) & ~isnan(ratio)));
critgam = median(critgam(theta_pos));
theta = mode(theta(theta~=10 & ~isnan(theta) & ~isnan(ratio)));
ratio = median(ratio(theta_pos & ratio~=0 & ~isnan(ratio)));

%% Plot the directional semivariograms
ranges = zeros(12,1);
for i = 1:ntheta
    for j = 1:10
        smoothSval = smoothdata(S_dir.val(:,i),'loess',10); 
        range_val = smoothSval >= critgam;
        range_ind = find(range_val==1,1,'first');
        if isempty(range_ind)
            critgam = .9*critgam;
        else
            break
        end
    end
    if range_ind == 1
        ranges(i) = S_dir.distance(range_ind);
    else
        ranges(i) = (S_dir.distance(range_ind)+S_dir.distance(range_ind-1))/2;
    end
    subplot(4,3,i)
    plot(S_dir.distance,S_dir.val(:,i),'o')
    axis([0 maxdist/2 0 max(max(S_dir.val))+.05*max(max(S_dir.val))])
    hold on
    plot(S_dir.distance,smoothSval)
    plot([0 ranges(i)], [critgam critgam],'k',[ranges(i) ranges(i)], [0 critgam],'k')
    hold off
    title(['\psi = ', num2str(xtheta(i)/pi*180),';  range = ',num2str(round(ranges(i),2))])
end

%% Compute omnidirectional semivariogram
% Determine which direction is that of max anisotropy and determine the
% range value for that direction and the one perpendicular to it.
% We define max anisotropy with max ratios, not which range is largest.

% Rotate the axes to get a new coordinate system.
% The major axis becomes the x-axis and the minor axis because the y-axis.
R = [cos(theta) sin(theta);
     -ratio*sin(theta) ratio*cos(theta)];
NewCoords =(R*xycoords')'; % New Coordinates
 
% Now adjust the distances accordingly. 
% Multiply the new y-corredinates by the ratio is the correlation length is
% the same in each direction. Then we can fit a new omnidirectional
% variogram to get the best l1 estimate. Then l2 = l1/ratio.
disp(' *** Computing omnidirectional variogram *** ')
maxdist = mean([sqrt(sum((max(NewCoords)-min(NewCoords)).^2)),maxdist]);
S = variogram([NewCoords(:,1) NewCoords(:,2)],vec,'nrbins',25,...
    'maxdist',maxdist/10,'subsample', Inf);
nu = var_opt_nu_disc(2,S,mean(ranges),max(S.val),min(S.val));
gam = variogramfit(S.distance,S.val,mean(ranges),max(S.val),S.num,'plotit',false,...
    'model','matern','nu',nu,'weightfun','cressie85','nugget',min(S.val));
l1 = gam;
l2 = gam/ratio;

% We now have estimates for l1, l2, theta, and nu