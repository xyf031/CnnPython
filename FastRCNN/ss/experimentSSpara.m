function experimentSSpara(sigma, singleK, minSize)
%20161129-Experiment: Search for best parameters for ss.
%  sigma = [0.7, 0.8, 0.9], singleK = [200, 500, 800, 1000, 1500, 2000], minSize = [200, 500, 800, 1000]

images = {'2.bmp'};
K = [singleK];
bbox = getSelectiveSearch(images, sigma, K, minSize);
% save bbox;
save(strcat(num2str(sigma), '-k', num2str(K(1)), '-ms', num2str(minSize), '.mat'), 'bbox');

im = imread(images{1});
rois = bbox{1};
roiN = size(rois, 1);
skip = 100;
roiCount = min(1000, roiN);
for i = 1:skip:roiCount
	iEnd = i + skip - 1;
	if iEnd > roiCount
		someBoxes = rois(i:roiCount, :);
	else
		someBoxes = rois(i:iEnd, :);
	end
	imNew = drawBox(im, someBoxes, 2);
	imwrite(imNew, strcat('bmp/', num2str(sigma), '-k', num2str(K(1)), '-ms', num2str(minSize), '=', num2str(i), '-', num2str(iEnd), '.bmp'));
end

