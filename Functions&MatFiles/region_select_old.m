function[binaryImage, nregions] = region_select_old(b,nm,fignum)
% This function creates masks for the different regions of an image for use
%   in inverse problems. This is compatible with all versions of MATLAB. If 
% The inputs are b, nm and fignum. b is the data from which we want to
%   specifiy the region. nm is the size of the original image (n x m) (the
%   default is twice the size of b). fignum is the figure number to be used
%   when viewing the image.
% This function also performs classification to assign any pixels not in a
%   region to the closest region.
% This function also assigns any overlapping of the regions to the eariler
%   region drawn.
if nargin < 3, fignum = 1; end
if nargin < 2, nm = 2*size(b); end
szb1 = size(b,1);
szb2 = size(b,2);
padsz1 = floor(szb1/10);
padsz2 = floor(szb2/10);
b_pad = padarray(b,[padsz1,padsz2]);
sz1 = size(b_pad,1);
sz2 = size(b_pad,2);
answer_draw = 1;
while answer_draw == 1
    if ishandle(fignum)
        close(fignum)
    end
    figure(fignum)
    imagesc(b_pad), axis image, colormap(gray)
    set(gca,'xticklabel',[])
    set(gca,'yticklabel',[])
    set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
    message = sprintf(['Left click and hold to begin drawing.'...
    '\nYou must create a closed loop.'...
    '\nDrawing outside the image is fine.'...
    '\nEach region can contain multiple sections.'...
    '\nClick the mouse after drawing each region.'...
    '\nPress Escape twice to draw a new region or to proceed to quit.']);
    uiwait(msgbox(message));
    nregions = 1;
    regparts = 1;
    binaryImage = cell(1,1);
    H = cell(1,1);
    % first level is for nregions, second level is for regparts
    H{1} = cell(1,1);
    H{1}{1} = imfreehand();
    totMask = false([sz1,sz2]);
    BWmask = H{1}{1}.createMask();
    if isvalid(H{1}{1}) && sum(BWmask(:)) > 10
        invalid_flag = 0;
        %pause;
        H{nregions}{regparts}.setColor('red');
        % less than 10 pixels is considered empty mask
        while sum(BWmask(:)) > 10
            totMask = totMask | BWmask; % add mask to global mask
            % ask user for another mask
            regparts = regparts + 1;
            H{nregions}{regparts} = imfreehand();
            %pause;
            BWmask = H{nregions}{regparts}.createMask();
            H{nregions}{regparts}.setColor('red');
        end
        binaryImage{nregions} = totMask;

        answer_more = 'Yes';
        while answer_more == 'Yes'
            answer_more = questdlg('Is there another distinct region?', ...
                'Regions', ...
                'Yes','No','No');
        % Handle response
            switch answer_more
                case 'Yes'
                    nregions = nregions + 1;
                    if nregions == 2
                        reg_color = 'yellow';
                    elseif nregions == 3
                        reg_color = 'green';
                    elseif nregions == 4
                        reg_color = 'magenta';
                    elseif nregions == 5
                        reg_color = 'cyan';
                    elseif nregions == 6
                        reg_color = 'white';
                    elseif nregions >= 7
                        reg_color = 'black';
                    end
                    regparts(nregions) = 1;
                    H{nregions}{regparts(nregions)} = imfreehand();
                    %pause;
                    H{nregions}{regparts(nregions)}.setColor(reg_color);
                    totMask = false([sz1,sz2]);
                    BWmask = H{nregions}{regparts(nregions)}.createMask();
                    % less than 10 pixels is considered empty mask
                    while sum(BWmask(:)) > 10
                        totMask = totMask | BWmask; % add mask to global mask
                        regparts(nregions) = regparts(nregions) + 1;
                        % ask user for another mask
                        H{nregions}{regparts(nregions)} = imfreehand();
                        %pause;
                        H{nregions}{regparts(nregions)}.setColor(reg_color);
                        BWmask = H{nregions}{regparts(nregions)}.createMask();
                    end
                    binaryImage{nregions} = totMask;
                case 'No'
                    break;
            end
        end
        answer_draw_text = questdlg('Would you like a redraw?', ...
            'Redraw', ...
            'Yes','No','No');
        % Handle response
        switch answer_draw_text
            case 'Yes'
                answer_draw = 1;
            case 'No'
                answer_draw = 0;
        end
    else
        pause;
        empty_ans = questdlg(['Since no region was drawn, the entire '...
            'image will be treated as a single region. '...
            'Is this okay?'], ...
                'No Region Drawn', ...
                'Yes','No','No');
        % Handle response
            switch empty_ans
                case 'Yes'
                   invalid_flag = 1;
                   answer_draw = 0; 
                case 'No'
                    answer_draw = 1;
            end
    end
end
if invalid_flag == 1
    binaryImage = cell(1,1);
    binaryImage{1} = ones(size(b));
else
    % Create all masks the same size as b. Creating the masks here at the
    % end allows each one to be adjusted until the user is finished.
    for i = 1:nregions
        binaryImage{i} = binaryImage{i}...
            (padsz1+1:end-padsz1,padsz2+1:end-padsz2);
    end
    
    % Create mask for any part of the image not included in other regions.
    binaryImage{nregions+1} = false(size(binaryImage{1}));
    for j = 1:nregions
        binaryImage{nregions+1} = binaryImage{nregions+1} | binaryImage{j};
    end
    nregions = nregions + 1;
    binaryImage{nregions} = binaryImage{nregions} == 0;
end
% Return a mask the size n x m.
for i = nregions:-1:1
    if sum(sum(binaryImage{i})) < 10 % removes any empty masks
        binaryImage(i) = [];
        nregions = nregions - 1;
    else
    binaryImage{i} = padarray(binaryImage{i},...
        [(nm(1)-szb1)/2,(nm(2)-szb2)/2],0);
    end
end
nregions = size(binaryImage,2); % make sure nregions aligns with number of masks
if nm(1)>szb1 || nm(2)>szb2
    % Begin Classification:
    N = nm(1)*nm(2);
    data_frame = zeros(N,nregions+4);
    data_frame(:,1) = 1:N;
    % Det up data frame with the element numbers and their xy coordinates
    xycoord = [repelem((1:nm(2))',nm(2)) repmat((nm(1):-1:1)',nm(1),1)];
    data_frame(:,2:3) = xycoord;
    % Add mask data to the data frame
    for region = 1:nregions
        data_frame(:,3+region) = binaryImage{region}(:);
    end
    % 
    data_frame(:,4+nregions) = sum(data_frame(:,4:3+nregions),2);
    data_frame_classified = data_frame(data_frame(:,4+nregions)>0,:);
    data_frame_not_class = data_frame(data_frame(:,4+nregions)==0,:);

    n_nc = size(data_frame_not_class,1); % size not classified
    % Classify every element of the extended mask into one of the regions
    % based on nearest neighbor. 
    IDX = knnsearch(data_frame_classified(:,2:3),data_frame_not_class(:,2:3));
    data_frame_not_class(:,4:4+nregions) = data_frame_classified(IDX,4:4+nregions);
    data_frame_new = sortrows([data_frame_classified; data_frame_not_class],1);
    for region = 1:nregions
        binaryImage{region} = logical(reshape(data_frame_new(:,3+region),nm(1),nm(2)));
    end
end

% Assign overlaping in regions to the region drawn eariler.
for i = 1:nregions-1
    for j=i+1:nregions
        if nnz(binaryImage{i}+binaryImage{j} > 1) > 0
        	binaryImage{j} = binaryImage{j}...
                - (binaryImage{i}+binaryImage{j} > 1);
        end
    end
end

set(gcf, 'Position',  [440 378 560 420]) % Return picture to default size