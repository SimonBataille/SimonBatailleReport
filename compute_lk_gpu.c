void OptFlowLK::compute_lk_gpu(Mat &frame1, Mat &frame2,
                               vector<Point2f> &features, vector<Point2f> &new_features,
                               vector<uchar> &status, int max_iterations, int cpt)
{
    /*****************************************************************
     *  Pre-processing
     *
     * **************************************************************/
    status.reserve(features.size());

    new_features.erase(new_features.begin(), new_features.end());
    new_features.clear();

    int nb_features = features.size();

    Mat frame1_float, frame2_float;

    frame1.convertTo(frame1_float, CV_32FC1, 1 / 255.0f);
    frame2.convertTo(frame2_float, CV_32FC1, 1 / 255.0f);

    float *frame1_float_ptr = (float *) frame1_float.data;
    float *frame2_float_ptr = (float *) frame2_float.data;

    unsigned int features_static[nb_features * 2];


    float d_pos[nb_features * 2];
    for (int i = 0; i < features.size(); i++)
    {
        d_pos[i] = 0.0f;
        d_pos[i + nb_features] = 0.0f;
    }


    for (int i = 0; i < features.size(); i++)
    {
        features_static[i] = (unsigned int)features.at(i).y;
        features_static[i + nb_features] = (unsigned int)features.at(i).x;
        status[i] = true;
    }



    /*****************************************************************
     *  GPU part
     *
     * **************************************************************/
    grad_dPosXYsharedInit();


    grad_firstFrameSharedCpy(frame1_float_ptr); //previous_gray
    grad_secondFrameSharedCpy(frame2_float_ptr); //gray
    grad_featureXYsharedCpy(features_static);


    grad_gradXqpu(0);
    grad_gradYqpu(0);
    grad_gradXYqpu();
    grad_detXYsharedCompute();


    for (int iteration = 0; iteration < max_iterations; iteration++)
    {
        grad_xtrBIP2qpu(0);
        grad_computeBIP2qpu(0);
        grad_dPosXYqpu(0);
    }

    grad_dPosXYsharedCpy(d_pos);



    /*****************************************************************
     *  Post-processing
     *
     * **************************************************************/
    for (int i = 0; i < features.size(); i++)
    {

        new_features.push_back(Point2f(features.at(i).x + d_pos[i + nb_features],
                                       features.at(i).y + d_pos[i]));
    }
