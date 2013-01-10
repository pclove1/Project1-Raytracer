numOfFrames = 30;
numOfControls = 5;
control_frames = round(linspace(1, 30, numOfControls));

%%
[control_X, control_Y] = ginput(numOfControls);
[control_Z, dummy] = ginput(numOfControls);

%%
control_X = rand(numOfControls, 1);
control_Y = rand(numOfControls, 1);
control_Z = rand(numOfControls, 1);

%% scale
control_X = control_X*4 - 2;
control_Y = control_Y*4;
control_Z = control_Z*3;

X = interp1(control_frames, control_X, 1:30, 'spline');
Y = interp1(control_frames, control_Y, 1:30, 'spline');
Z = interp1(control_frames, control_Z, 1:30, 'spline');

plot3(X, Y, Z, '*');

%%
for i = 1:numOfFrames
    fprintf('frame %d\n', i-1);
    fprintf('TRANS       %f %f %f\n', X(i), Y(i), Z(i));
    fprintf('ROTAT       0 180 0\n');
    fprintf('SCALE       3 3 3\n');
end

%% a fixed camera
for i = 1:numOfFrames
    fprintf('frame %d\n', i-1);
    fprintf('EYE         0 4.5 12\n');
    fprintf('VIEW        0 0 -1\n');
    fprintf('UP          0 1 0\n');
end

%% a fixed object
for i = 1:numOfFrames
    fprintf('frame %d\n', i-1);
    fprintf('TRANS       0 10 0\n');
    fprintf('ROTAT       0 0 90\n');
    fprintf('SCALE       .3 3 3\n');
end

%% 
addpath('S:\fall2012\CIS565(GPU)\Project1-Raytracer\PROJ1_WIN\Release\renders');

writerObj = VideoWriter('video.avi');
writerObj.FrameRate = 1;

open(writerObj);

for n = 1:30
   fname = sprintf('sampleScene.%d.bmp', n-1);
   frame = imread(fname);
   writeVideo(writerObj, frame);
end

close(writerObj);