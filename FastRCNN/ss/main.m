
im = imread();
bb = boxes{1};

rois = size(bb, 1);
skip = 10;

for i = 41:50
	bb1=bb(i, :);
	im1=drawBox(im, bb1, 2);
	imwrite(im1, strcat('box-', num2str(i), '.bmp'));
end

for i = 1: skip: rois
	if i + skip - 1 > rois
		break
	end
	bb1 = bbox(i:(i + skip - 1), :);
	im1 = drawBox(im3, bb1, 1);
	imwrite(im1, strcat('boxes-', num2str(i), '.bmp'));
end

im3 = imread(ims{1});
bbox=getSelectiveSearch(ims, 0.8, [200], 200);
% SS-Elapsed time: 1152.363353 seconds

bb1 = bbox{1}(1:100, :);
im1 = drawBox(im3, bb1, 2);
imwrite(im1, 'aaa.bmp');
