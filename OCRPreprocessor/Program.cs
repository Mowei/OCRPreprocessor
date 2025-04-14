using OpenCvSharp;
using System.Security.Cryptography;
using static System.Net.Mime.MediaTypeNames;


class OCRPreprocessor
{
    public static Mat PreprocessImage(string path)
    {
        Mat src = Cv2.ImRead(path, ImreadModes.Grayscale);

        // 中值濾波去除浮水印點狀紋理
        Mat blurred = new Mat();
        Cv2.MedianBlur(src, blurred, 3);

        // 自動二值化
        Mat binary = new Mat();
        Cv2.AdaptiveThreshold(blurred, binary, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 15, 10);

        return binary;
    }
    //XX 背景移除專用處理（形態學操作）
    public static Mat fun1(string path)
    {
        Mat src = Cv2.ImRead(path, ImreadModes.Grayscale);

        // 高斯模糊去雜訊 ---
        Mat blurred = new Mat();
        Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);

        //自動二值化
        Mat binary = new Mat();
        Cv2.AdaptiveThreshold(blurred, binary, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 15, 10);

        {

            // 找輪廓
            Cv2.FindContours(binary.Clone(), out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                Rect bbox = Cv2.BoundingRect(contour);
                if (area < 5 && IsSurroundedEndByWhite(binary, bbox))
                {
                    Cv2.DrawContours(binary, new[] { contour }, -1, Scalar.Black, -1); // 填成白色（去除小點）
                }
            }
        }
        
        //清除段落
        {
            Cv2.FindContours(binary.Clone(), out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                Rect bbox = Cv2.BoundingRect(contour);
                if (area < 10 && IsSurroundedByWhite(binary, bbox))
                {
                    Cv2.DrawContours(binary, new[] { contour }, -1, Scalar.Black, -1); // 填成黑色（刪除點）
                }
            }
        }



        return binary;
    }



    static Mat IsSideAlmostWhite(Mat binary)
    {

        int height = binary.Rows;
        int width = binary.Cols;

        int lineHeight = 20; // 每行高度預估（可調）

        for (int y = 0; y < height; y += lineHeight)
        {
            int yEnd = Math.Min(y + lineHeight, height);
            int whitePixelCount = 0;
            int totalPixels = (yEnd - y) * width;

            // 統計區塊內的白色像素數
            for (int i = y; i < yEnd; i++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (binary.At<byte>(i, x) < 5)
                        whitePixelCount++;
                }
            }

            double whiteRatio = (double)whitePixelCount / totalPixels;

            // 若此區段幾乎為白（空白行），則把這一整段塗白
            if (whiteRatio > 0.98)  // 閾值可調
            {
                for (int i = y; i < yEnd; i++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        binary.Set<byte>(i, x, 0); // 填白
                    }
                }
            }
        }
        return binary;
    }
    static bool IsSurroundedByWhite(Mat binary, Rect bbox)
    {
        int height = binary.Rows;
        int width = binary.Cols;

        int lineHeight = 5; // 每行高度預估（可調）

        int totalPixels = binary.Cols;

        int whitePixelCount = 0;

        int yEnd = Math.Min(bbox.Y + lineHeight, height);
        // 統計區塊內的白色像素數
        /*
        for (int x = 0; x < 600; x++)
        {
            for (int y = bbox.Y; y < yEnd; y++)
            {
                if (binary.At<byte>(y, x) < 5)
                    whitePixelCount++;
            }
        }
        */
        for (int x = 0; x < width; x++)
        {
            for (int y = bbox.Y; y < yEnd; y++)
            {
                if (binary.At<byte>(y, x) < 5)
                    whitePixelCount++;
            }
        }
        double whiteRatio = (double)whitePixelCount / (width * lineHeight);

        if (whiteRatio > 0.90)  // 閾值可調
        {
            return true;
        }
        return false;
    }

    static bool IsSurroundedEndByWhite(Mat binary, Rect bbox)
    {
        int height = binary.Rows;
        int width = binary.Cols;

        int lineHeight = 20; // 每行高度預估（可調）

        int totalPixels = binary.Cols;

        int whitePixelCount = 0;

        int yEnd = Math.Min(bbox.Y + lineHeight, height);
        // 統計區塊內的白色像素數

        for (int x = bbox.X; x < width; x++)
        {
            for (int y = bbox.Y; y < yEnd; y++)
            {
                if (binary.At<byte>(y, x) < 5)
                    whitePixelCount++;
            }
        }
        double whiteRatio = (double)whitePixelCount / ((width - bbox.X) * lineHeight);

        if (whiteRatio > 0.95)  // 閾值可調
        {
            return true;
        }
        return false;
    }

    static Mat RemoveBackground(string path)
    {
        Mat gray = Cv2.ImRead(path, ImreadModes.Grayscale);

        for (int y = 0; y < gray.Rows; y++)
        {
            for (int x = 0; x < gray.Cols; x++)
            {
                byte pixel = gray.At<byte>(y, x);
                if (pixel > 200) // 去除淺灰
                    gray.Set<byte>(y, x, 255); // 改成白色
            }
        }

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
        string inputDir = @"C:\Users\182798\Desktop\ch2_convertedToImages\Cut";    // 圖片來源資料夾
        string outputDir = @"C:\Users\182798\Desktop\ch2_convertedToImages\Pre";  // 處理後輸出資料夾

        if (!Directory.Exists(outputDir))
            Directory.CreateDirectory(outputDir);

        string[] files = Directory.GetFiles(inputDir, "*.*", SearchOption.TopDirectoryOnly);

        foreach (string file in files)
        {
            if (!file.EndsWith(".jpg") && !file.EndsWith(".png")) continue;

            Console.WriteLine($"處理中：{Path.GetFileName(file)}");

            Mat binary = new Mat();
            //binary = RemoveBackground(file);
            binary = fun1(file);

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
