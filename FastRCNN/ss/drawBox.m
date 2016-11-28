function im = drawBox(im, boxes, linewidth)
%Draw boxes in blue, and return im.
%	size(boxes) should be [r 4], and the columns must be [row1 col1 row2 col2];

boxSize = size(boxes);
if boxSize(2) ~= 4
	disp('ERROR: drawBox(im, boxes, linewidth=10). The columns of boxes is not 4.')
	im;
	return
end
boxes = uint16(boxes);

imSize = size(im);
smallSide = min(imSize(1), imSize(2));
linewidth = max(2, min(smallSide/100, linewidth));
move = round(linewidth / 2);
rowBound = imSize(1);
colBound = imSize(2);

for i = 1: boxSize(1)
	box = boxes(i, :);
	row1 = box(1);
	col1 = box(2);
	row2 = box(3);
	col2 = box(4);

	outrow1 = max(row1 - move, 1);
	outrow2 = min(row2 + move, rowBound);
	outcol1 = max(col1 - move, 1);
	outcol2 = min(col2 + move, colBound);
	inrow1 = min(row1 + move, rowBound);
	inrow2 = max(row2 - move, 1);
	incol1 = min(row1 + move, colBound);
	incol2 = max(row2 - move, 1);
	if outrow1 >= outrow2 | outcol1 >= outcol2 | inrow1 >= inrow2 | incol1 >= incol2 | (outrow2-outrow1) <= (inrow2-inrow1) | (outcol2-outcol1) <= (incol2-incol1)
		continue
	end

	boxIndex = false(imSize);
	boxIndex(outrow1:outrow2, outcol1:outcol2, 1:2) = 1;
	boxIndex(inrow1:inrow2, incol1:incol2, 1:2) = 0;
	im(boxIndex) = 0;
end

im;
