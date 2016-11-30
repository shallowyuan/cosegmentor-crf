% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
%%
function internet
    addpath('Dependencies');

    fprintf('Demo of how to run the code for:\n');
    fprintf('   J. Uijlings, K. van de Sande, T. Gevers, A. Smeulders\n');
    fprintf('   Segmentation as Selective Search for Object Recognition\n');
    fprintf('   IJCV 2013\n\n');

    % Compile anisotropic gaussian filter
    if(~exist('anigauss'))
        fprintf('Compiling the anisotropic gauss filtering of:\n');
        fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
        fprintf('   Fast anisotropic gauss filtering\n');
        fprintf('   IEEE Transactions on Image Processing, 2003\n');
        fprintf('Source code/Project page:\n');
        fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
        mex Dependencies/anigaussm/anigauss_mex.c Dependencies/anigaussm/anigauss.c -output anigauss
    end

    if(~exist('mexCountWordsIndex'))
        mex Dependencies/mexCountWordsIndex.cpp
    end

    % Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
    if(~exist('mexFelzenSegmentIndex'))
        fprintf('Compiling the segmentation algorithm of:\n');
        fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
        fprintf('   Efficient Graph-Based Image Segmentation\n');
        fprintf('   International Journal of Computer Vision, 2004\n');
        fprintf('Source code/Project page:\n');
        fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
        fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
    %     fprintf('   
        mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
    end

    minBoxWidth = 20;
    % Test the boxes
    basedir = '/home/zehuany/cosegmentor/ObjectDiscovery/Data/';
    fprintf('After box extraction, boxes smaller than %d pixels will be removed\n', minBoxWidth);
    fprintf('Obtaining boxes for Internet set:\n');
    totalTime = 0;

    datasets= dir(basedir);
    for m=1:length(datasets)
        if strcmp(datasets(m).name,'.') || strcmp(datasets(m).name,'..')
            continue;
        end
        if datasets(m).isdir
            tic 
            getBoxes(basedir,[datasets(m).name '/'],minBoxWidth);
            totalTime = totalTime + toc;
            fprintf('Time for Dataset %s: %.2f ...\n', datasets(m).name,totalTime);
        end
    end
    fprintf('\n');
end
function getBoxes(basedir,folder,minBoxWidth)
    %%
    % Parameters. Note that this controls the number of hierarchical
    % segmentations which are combined.
    colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};

    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
    sigma = 0.8;

    % After segmentation, filter out boxes which have a width/height smaller
    % than minBoxWidth (default = 20 pixels).


    % Comment the following three lines for the 'quality' version
    % colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
    % simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
    % ks = ks(1:2);

    cnames=dir([basedir folder]);
    totalTime=0;
    pdir=[basedir folder 'Proposals'];
    for i=1:length(cnames)
        if strcmp(cnames(i).name,'.' )|| strcmp(cnames(i).name,'..') || strcmp(cnames(i).name,'GroundTruth')
            continue;
        end
        if cnames(i).isdir
            getBoxes([folder cnames(i).name '/']);
        end
        fprintf('proceed %s \n', [folder cnames(i).name]);
        % VOCopts.img
        im = imread([basedir folder cnames(i).name]);
        if size(im,3)==1
            im=cat(3, im, im, im);
        end
        idx = 1;
        for j=1:length(ks)
            k = ks(j); % Segmentation threshold k
            minSize = k; % We set minSize = k
            for n = 1:length(colorTypes)
                colorType = colorTypes{n};
                tic;
                [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
                totalTime = totalTime + toc;
                idx = idx + 1;
            end
        end
        boxes = cat(1, boxesT{:}); % Concatenate boxes from all hierarchies
        priority = cat(1, priorityT{:}); % Concatenate priorities
        
        % Do pseudo random sorting as in paper
        priority = priority .* rand(size(priority));
        [priority sortIds] = sort(priority, 'ascend');
        boxes = boxes(sortIds,:);
        boxes = FilterBoxesWidth(boxes, minBoxWidth);
        boxes = BoxRemoveDuplicates(boxes);
        if ~exist(pdir,'dir')
            mkdir(pdir)
        end
        save(fullfile(pdir,[cnames(i).name(1:end-4) '.mat']),'boxes');
    end
end
