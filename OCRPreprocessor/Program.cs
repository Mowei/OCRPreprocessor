using OpenCvSharp;


class OCRPreprocessor
{
    static Mat RemoveBackground(Mat gray)
    {
        // 轉灰階
        //Mat gray = new Mat();
        //Cv2.CvtColor(gray, gray, ColorConversionCodes.BGR2GRAY);

        //銳化字體（可選）：
        //Mat fonted = new Mat();
        //Cv2.Laplacian(gray, fonted, MatType.CV_8U);

        // 高斯模糊去雜訊 ---
        Mat blurred = new Mat();
        Cv2.GaussianBlur(gray, blurred, new Size(5, 5), 0);

        /*
        // 估計背景（用模糊模擬背景）
        Mat background = new Mat();
        Cv2.GaussianBlur(fonted, background, new Size(25, 25), 0);

        // 背景減法
        Mat diff = new Mat();
        Cv2.Absdiff(fonted, background, diff);

        // 對比拉伸
        Mat norm = new Mat();
        Cv2.Normalize(diff, norm, 0, 255, NormTypes.MinMax);



        //膨脹 + 開運算 去除雜點
        Mat cleaned = new Mat();
        Mat kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
        //Cv2.MorphologyEx(norm, cleaned, MorphTypes.Open, kernel);
        */

        // 二值化
        Mat binary = new Mat();
        //Cv2.Threshold(blurred, binary, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        //自動二值化
        Cv2.AdaptiveThreshold(blurred, binary, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 15, 10);

        return binary;
    }

    static void Main(string[] args)
    {
        string inputDir = @"C:\Users\182798\Desktop\ch3_convertedToImages\Cut";    // 圖片來源資料夾
        string outputDir = @"C:\Users\182798\Desktop\ch3_convertedToImages\Pre";  // 處理後輸出資料夾

        if (!Directory.Exists(outputDir))
            Directory.CreateDirectory(outputDir);

        string[] files = Directory.GetFiles(inputDir, "*.*", SearchOption.TopDirectoryOnly);

        foreach (string file in files)
        {
            if (!file.EndsWith(".jpg") && !file.EndsWith(".png")) continue;

            Console.WriteLine($"處理中：{Path.GetFileName(file)}");

            Mat src = Cv2.ImRead(file, ImreadModes.Grayscale);


            Mat binary = new Mat();
            binary = RemoveBackground(src);

            // --- 1. 高斯模糊去雜訊 ---
            //Mat blurred = new Mat();
            //Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);

            // --- 2. 自動二值化 ---
            //Mat binary = new Mat();
            //Cv2.AdaptiveThreshold(blurred, binary, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 15, 10);

            // --- 3. 傾斜矯正 ---
            //Mat deskewed = Deskew(binary);

            // --- 4. 儲存 ---
            string outputPath = Path.Combine(outputDir, Path.GetFileName(file));
            Cv2.ImWrite(outputPath, binary);
        }

        Console.WriteLine("全部圖片處理完成！");
    }

    // --- 傾斜矯正函式 ---
    static Mat Deskew(Mat src)
    {
        // 找輪廓
        Point[][] contours;
        HierarchyIndex[] hierarchy;
        Cv2.FindContours(src, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        RotatedRect largestBox = new RotatedRect();
        double maxArea = 0;

        foreach (var contour in contours)
        {
            var box = Cv2.MinAreaRect(contour);
            double area = box.Size.Width * box.Size.Height;
            if (area > maxArea)
            {
                maxArea = area;
                largestBox = box;
            }
        }

        double angle = largestBox.Angle;
        if (largestBox.Size.Width < largestBox.Size.Height)
            angle = angle + 90;

        // 旋轉矯正
        Point2f center = new Point2f(src.Width / 2, src.Height / 2);
        Mat rotMat = Cv2.GetRotationMatrix2D(center, angle, 1.0);
        Mat rotated = new Mat();
        Cv2.WarpAffine(src, rotated, rotMat, src.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.White);

        return rotated;
    }
}
