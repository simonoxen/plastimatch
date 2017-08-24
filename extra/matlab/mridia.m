function pos = mridia (img, roi)

img_roi = img(roi(1):roi(2),roi(3):roi(4));
img_roi = mean(img_roi,2);
gradient = diff(img_roi);
gavg = gradient(1:end-1) + gradient(2:end);
[gavg_min,min_loc] = min(gavg);
pos = roi(1) + min_loc + 1;

