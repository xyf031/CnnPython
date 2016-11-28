function boxes = getSelectiveSearch(images, sigma, kThresholds, minSize)
%SelectiveSearch   images = {'/home/1.bmp', '/home/2.bmp'};  sigma = 0.8;  kThresholds = [100 200];
%    return boxes{};  boxes{i} = rois * 4;  boxes{i}[j, :] = [rowBegin, colBegin, rowEnd, colEnd];

colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};
numHierarchy = length(colorTypes) * length(kThresholds);
boxes = cell(1, length(images));
tic;
for i = 1: length(images)
    if mod(i, 100) == 0
        fprintf('%d ', i);
    end

    idx = 1;
    currBox = cell(1, numHierarchy);
    im = imread(images{i});
    for k = kThresholds
        %minSize = k;
        for colorTypeI = 1: length(colorTypes)
            colorType = colorTypes{colorTypeI};
            currBox{idx} = SelectiveSearch(im, sigma, k, minSize, colorType);
            idx = idx + 1;
        end
    end
    
    boxes{i} = cat(1, currBox{:});
    boxes{i} = unique(boxes{i}, 'rows');
end
fprintf('SS-Elapsed time: %f seconds\n', toc);
boxes;
