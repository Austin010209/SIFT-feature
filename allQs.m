clc;
clear;
close all;
rng(5);  %when running the code, you can delete it

im = imread('p2.jfif');    %p1.jpg for another image
im = rgb2gray(im);        %convert the image to gray
im = imresize(im,1.2);     %mainly used in Q7
scales = zeros(size(im,1),size(im,2),17);    %scale space, 17 layers
sigma0 = 1;              %initial sigma
step = 2^(1/4);          %change of sigma in each layer
for i=0:16
    fun = fspecial('gaussian', 20, sigma0 * step^i);  %construct Gaussian with specified sigma
    scales(:,:,i+1) = conv2(im, fun, 'same');   %convolve image with gaussian
end

% Q1
figure();
for i=1:16
    subplot(4,4,i);
    imagesc(scales(:,:,i+1));      %draw each layer
    title('sigma is ' + string(sigma0 * step^i));
end
colormap(gray);     %use gray colormap
sgtitle('Q1 Gaussian Scale Space');     %the title for all subplots
colorbar('Position', [0.92 0.1 0.03 0.82]);   %colorbar for all pictures







%Q2
figure();
for i=1:16
    subplot(4,4,i);
    sigma = sigma0 * step^i;
    plotone(sigma,scales(:,:,i+1));     %calling the plotone function to plot the map of harris-steven values
    title('sigma is ' + string(sigma));
end
sgtitle('Q2 Harris-Stevens');
%the color bar is different, so we have a color bar for each






% Q3
DOGs = zeros(size(scales,1), size(scales,2), 16);   %create DOG space
fun = fspecial('log',20,1);    %for zero_crossing_detection
figure();
sgtitle('Q3 DOGs');
for i = 1 : 16
    DOGs(:,:,i) = scales(:,:,i+1)-scales(:,:,i);     %DOG is just a subtraction between layers
    thezeros = zero_crossing_detector(DOGs(:,:,i),fun);
    subplot(4,4,i);
    imagesc(DOGs(:,:,i));
    %imagesc(thezeros);
    title('sigma is ' + string(sigma0 * step^i));
end
colormap(jet);
colorbar('Position', [0.92 0.1 0.03 0.82]);







% Q4
figure();
imagesc(im);
title('Q4 SIFT Keypoint Detection');
colormap(gray);
colorbar;
%we will draw circles on top of the image


keypoints = zeros(5000,4);    %pre-allocate the keypoints
curpos = 1;    %current_position: for adding new keypoints to the array of keypoints to the correct position(next avaliable position)
for layer=2:15    %we only search through layers 2 to 15
    sigma = round(sigma0 * step^layer);
    x = 2*sigma+1 : size(im, 1)-2*sigma;    %stay away from 2*sigma (we need add 1 to the start index, because start index of matlab is 1, not 0)
    y = 2*sigma+1 : size(im, 2)-2*sigma;
    [X, Y] = meshgrid(x,y);
    pairs = [X(:) , Y(:)];   %generate all pairs of (x,y) in the valid part of image
    for i = 1:size(pairs,1)   %for every pair
        x0 = pairs(i,1);    %x of the pair
        y0 = pairs(i,2);    %y of the pair
        thepoint = DOGs(x0, y0, layer);
        neighborhood = DOGs(x0-1:x0+1, y0-1:y0+1, layer-1:layer+1);   %neighborhood around the point
        neighbormax = max(neighborhood, [], 'all');    %maximum in the neighborhood, element-wise
        neighbormin = min(neighborhood, [], 'all');    %minimum in the neighborhood, element-wise
        dif = neighborhood - thepoint;        %the difference of all points in the neighborhood with the center
        ls = dif(dif>-2*eps & dif<2*eps);     %ls is the points that are essentially the same as the center
        %the center is max/min  &&  it is larger or smaller than some threshold  &&  make sure no other points is the same as the center point if length of ls is 1
        if ((thepoint > 3 && thepoint == neighbormax) || (thepoint < -3 && thepoint == neighbormin) && numel(ls)==1 )
            keypoints(curpos,1:3) = [x0, y0, layer];    %add the point to the keypoints
            curpos = curpos + 1;    %increment position (like i++)
        end
    end
end
keypoints = keypoints(keypoints(:,3) ~= 0,:);  %eliminate the all empty rows in the bottom (we choose the third column, because scale cannot be 0; but in this case, other two columns will work as well)
for i=1:size(keypoints,1)
     viscircles([keypoints(i,2), keypoints(i,1)], keypoints(i,3));   %draw the circle for each keypoints
end










%Q5
figure();
imagesc(im);
title('Q5 Hessian constraint');
colormap(gray);
colorbar;

newkps = zeros(5000,4);
curpo = 1;
xdrfilter = [1 0 -1; 1 0 -1; 1 0 -1]/6;     %derivative filter
ydrfilter = [1 0 -1; 1 0 -1; 1 0 -1]'/6;
for i = 1:size(keypoints, 1)     %for every keypoints generated before
    x0 = keypoints(i,1);
    y0 = keypoints(i,2);
    sc = keypoints(i,3);
    neighbors = DOGs(x0-2:x0+2, y0-2:y0+2, sc);   %we need a 5*5 neighbor to approximate hessian
    
    xdr = conv2(neighbors,xdrfilter,'valid');   %the derivative wrt x, resulting in 3*3
    ydr = conv2(neighbors,ydrfilter,'valid');   %%the derivative wrt y, resulting in 3*3
    xxdr = conv2(xdr,xdrfilter,'valid');        %second derivative wrt x, resulting in scalar
    xydr = conv2(xdr,ydrfilter,'valid');        %derivative of x * derivative of y, resulting in scalar
    yydr = conv2(ydr,ydrfilter,'valid');        %second derivative wrt y, resulting in scalar
    hessian = [xxdr, xydr; xydr, yydr];         %construct hessian
    r = 0.5;       %it is the same effect of 1/r (1/0.5) by math
    threshold = (r+1)^2/r;
    if (trace(hessian)^2/det(hessian) < threshold && det(hessian)>0)    %by the definition of Lowe's condition
        newkps(curpo,:) = keypoints(i,:);     %adding key points
        curpo = curpo + 1;
    end
end
newkps = newkps(newkps(:,3) ~= 0,:); 

for i=1:size(keypoints,1)
     viscircles([keypoints(i,2), keypoints(i,1)], keypoints(i,3));   %draw the circle for each keypoints
end
for i=1:size(newkps,1)
    viscircles([newkps(i,2), newkps(i,1)], newkps(i,3), 'color', 'b');    %draw blue circles
end







%Q6

figure();
imagesc(im);
title('Q6 SIFT Feature Dominant Orientation');
colormap(gray);
colorbar;

binhists = zeros(300, 36);
Newkps = zeros(2000,4);
curpo = 1;    %one index for adding new key points
curpo2 = 1;   %one index for adding new histograms (an array of numbers)
xdrfilter = [1 0 -1; 1 0 -1; 1 0 -1]/6;
ydrfilter = [1 0 -1; 1 0 -1; 1 0 -1]'/6;
hist = zeros(1,36);
for i = 1:size(newkps, 1)    %for every keypoint generated in Q5
    x0 = newkps(i,1);
    y0 = newkps(i,2);
    sc = newkps(i,3);
    truesigma = sigma0 * step^sc;
    sigma = round(truesigma);
    
    %neighbor is a square with width sigma+1 and centered in x0 y0 in the same scale
    neighbors = scales(x0-sigma-1 : x0+sigma+1,  y0-sigma-1 : y0+sigma+1, sc);
    
    xdr = conv2(neighbors,xdrfilter,'valid');
    ydr = conv2(neighbors,ydrfilter,'valid');
    mag = sqrt(xdr.^2 + ydr.^2);   %compute magnitude of gradient
    gaussian = fspecial('gaussian', 2*sigma+1, 1.5*truesigma);  %make the size of gaussian the same(compatible)
    weightedmag = mag.*gaussian;    %weight the gaussian wrt the neighborhood
    
    
    dir = atan2d(-xdr, ydr);        %calculate direction
    dir(dir<0) = dir(dir<0)+360;    %since it returns [-180,180], for convinience we convert it to [0,360]
    bin = round(dir/10)+1;          %compute the corresponding bin
    bin(bin==37) = 1;               %then there will be bin 37; it really should be 1
    
    binhist = zeros(1,36);
    for j = 1:numel(weightedmag)    %or the number of point in neighborhood
        %bin is the direction for each point in the neighborhood
        %we make it a linear index, so bin(j) is the direction of each
        %point
        %so we update the corresponding bin(wrt the bin(j)) in binhist by
        %weighted magnitude.
        binhist(bin(j)) = binhist(bin(j)) + weightedmag(j);
    end
    %now the histogram for each keypoint is done; we "append" it to binhists
    binhists(curpo2,:) = binhist;
    curpo2 = curpo2+1;
    
    %select the ones that have a height more than 80% of max
    larges = binhist( binhist > 0.8*max(binhist))';
    for k=1:length(larges)  %for each of the large keypoints
        ind = find(binhist == larges(k));   %find the index(kth bin)
        angle = (ind-1)*10;                 %find the angle according to bin
        Newkps(curpo,:) = [x0 y0 sc angle]; %append keypoint with angle
        curpo = curpo + 1;
        
        %now draw line
        %since angle in matlab is clockwise, and we want to use the convention(counterclockwise), so we make a minus sign for angle
        deltax = truesigma*2*cos(-1*deg2rad(angle));
        deltay = truesigma*2*sin(-1*deg2rad(angle));
        %so the incrementation is all plus. Draw the line
        line([y0 y0+deltay], [x0 x0+deltax], 'Color', 'r', 'LineWidth', 2);
    end
end
Newkps = Newkps(Newkps(:,3) ~= 0,:); 
binhists = binhists(sum(~binhists,2) ~= size(binhists,2) ,:);
%truncate the zeros of keypoint list and binhist list


%draw histograms
twopoints = randi([1 size(binhists,1)],2,1);  %randomly select 2 points within binhists
for i = 1:2
    figure();
    thehist = binhists(twopoints(i),:);  %select the binhists according to the result
    x = 0:1:35;
    bar(x,thehist);  %draw a bar graph(we do not use hist because the histogram is already converted into a list)
    xlabel('degrees');
    ylabel('counts');
    title(sprintf('histogram of point  y(row) = %d   x(col) = %d', newkps(twopoints(i),1), newkps(twopoints(i),2)));
    xticks(0:35);
    xticklabels(0:10:350);  %make the x show 0,10...350, not 0,1,...35
end