
% run main.m
% images = {'/home/xyf/ssd/newMRC/gammas-lowpass/stack_0001_cor.mrc.bmp'};

images = {'2.bmp'};
sigma = 0.8;
K = [1000];
minSize = 200;
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
	imwrite(imNew, strcat('bmp/boxes-', num2str(i), '-', num2str(iEnd), '.bmp'));
end


%%%%%%%%%%
for sigma = [0.7, 0.8, 0.9]
	for singleK = [200, 500, 800, 1000, 1500, 2000]
		for minSize = [200, 500, 800, 1000]
			experimentSSpara(sigma, singleK, minSize);
		end
	end
end


%%%%%%%%%%%%%%
for i = 41:50
	bb1=bb(i, :);
	im1=drawBox(im, bb1, 2);
	imwrite(im1, strcat('box-', num2str(i), '.bmp'));
end



%%%%%%%%%%%%%
sigma = 0.8;
K = [1000];
minSize = 200;
SS-Elapsed time: 597.449118 seconds
roiN = 4178
