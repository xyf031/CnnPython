function experi()
%Experiment on 20161129
%  Search for the best PARAMETER of Selective Search
%  Generate BMP for the first 1000 boxes, and pick the best by person.

sigma = 0.8;
for singleK = [200, 500, 1000, 1500, 2000]
	for minSize = [200, 500, 800, 1000]
		experimentSSpara(sigma, singleK, minSize);
	end
end

sigma = 0.7;
for singleK = [500, 1000, 1500, 2000]
	for minSize = [200, 500, 800]
		experimentSSpara(sigma, singleK, minSize);
	end
end

sigma = 0.9;
for singleK = [500, 1000, 1500, 2000]
	for minSize = [200, 500, 800]
		experimentSSpara(sigma, singleK, minSize);
	end
end
