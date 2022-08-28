function [] = plotone(sig, im)
    [dx,dy] = imgradientxy(im); %compute image gradient
    dx2 = dx.*dx;   %compute square of 1st derivative
    dy2 = dy.*dy;   %compute square of 1st derivative
    dxy = dx.*dy;   %compute product of 2 1st derivatives

    fun = fspecial('gaussian',6*floor(sig),2*sig);
    %compute x2,y2 and xy for all points
    x2 = conv2(dx2,fun);
    y2 = conv2(dy2,fun);
    xy = conv2(dxy,fun);

    row = size(im,1);
    col = size(im,2);
    %pre allocate the image
    newim = zeros(row,col);
    for x = 1:col
        for y = 1:row
            %construct hessian
            M = [x2(y,x), xy(y,x) ; xy(y,x), y2(y,x)];
            %compute harris-steven value
            newim(y,x) = det(M) - 0.1*trace(M)^2;
        end
    end
    
    %draw the image
    imagesc(newim);
    colormap(jet);
    colorbar;
end