% =========================================================================
% function DrawTrackedTemplate(nTemplates, templateParams)
% 
% Draw the templates on the source image based on the estestimated
% parameters like location, scale and rotation
% 
% Author: Rui Li
% Date: 09/17/2009
% =========================================================================
function DrawTrackedTemplate(nTemplates, templateParams)
gcf; hold on;
colorSpecs = ['y','m','c','r','g','b','w','k'];
for iTemplate = 1:nTemplates
    % the size and rotation angle of the template
    tLength = templateParams(iTemplate, 1);
    tHeight = templateParams(iTemplate, 2);
    tAngle = templateParams(iTemplate,3);
    
    % center in the source image coordinate system
    tRow = templateParams(iTemplate,4);
    tCol = templateParams(iTemplate,5);
    
    DrawRectangle(tLength, tHeight, tAngle, tRow, tCol, ...
        colorSpecs(iTemplate));
end

hold off;

% rLength, rHeight: the long and short sides of the rectangle
% rAngle: the angle of rotation of the rectangle
% x0, y0: the center of the rectangle
% rColor: the color of the lines of the rectangle
function DrawRectangle(rLength, rHeight, rAngle, y0, x0, rColor)

% add some margin to the tight template width and height
% for better visualization
xMargin = 8; 
yMargin = 4;
rLength = rLength + xMargin;
rHeight = rHeight + yMargin;

% the 4 corners of the rectangle
xn = [-rLength rLength rLength -rLength];
yn = [-rHeight -rHeight rHeight rHeight];

% rotation matrix
rot = [cos(rAngle), sin(rAngle); ...
    -sin(rAngle), cos(rAngle)];

% rotate rectangle
xy = rot * [xn; yn];

x = xy(1,:) + x0;
y = xy(2,:) + y0;
plot(x([1 2 3 4 1]),y([1 2 3 4 1]),[rColor,'-'], 'LineWidth', 2);


