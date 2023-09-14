% Load the detector.

%30 epoch
load('30128cwSignsDetector.mat', 'detector');

% Define the test image file names.
input_ims = {'02.jpg','03.jpg', '01.jpg','04.jpg', '05.jpg','06.jpg','07.jpg', '08.jpg'};

% Loop through the test images and detect objects in each image.
for i = 1:length(input_ims)

    % Read the current test image.
    input_im = imread(input_ims{i});
    In_mean = mean2(input_im);

    if In_mean < 100
        disp('This is a night image.');
        %increase contrast of the image
        %input_im = imadjust(input_im,[],[],9);
        input_im = imadjust(input_im,[],[],5);
        %input_im = imadjust(input_im,[],[],12);
    else
        disp('This is a day image.');
        
        
        % Histogram equalization
        input_im = histeq(input_im);
        % decrease white intensity of image


        input_im = imadjust(input_im,[],[],0.60);

        %input_im = imadjust(input_im,[],[],0.499);
        
    end
    % Detect objects in the current test image.
    [bboxes, probability, labels] = detect(detector, input_im);

    threshold_boolean = probability > 0.999;

    thresholdBboxes = bboxes(threshold_boolean, :);
    thresholdprobability = probability(threshold_boolean, :);
    thresholdLabels = labels(threshold_boolean, :);
    label_str = cellstr(thresholdLabels);

    % Display the detection results in a new figure.
    figure();
    if isempty(thresholdprobability)
        disp(['No objects detected in ' input_ims{i} '.']);
        imshow(input_ims{i});
    else
        outputImage = insertObjectAnnotation(input_im, 'rectangle', thresholdBboxes, thresholdprobability);
        imshow(outputImage);
    end

end
