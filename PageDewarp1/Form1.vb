
Imports Emgu.CV
Imports Emgu.CV.Util
Imports Emgu.CV.CvEnum
Imports Emgu.CV.Structure
Imports System.Linq
Imports System.Collections.Generic

Public Class Form1

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load

    End Sub

    Private Sub Button1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button1.Click

        Dim RVec As New Mat
        Dim TVec As New Mat

        'zer5 = np.zeros(5)
        Dim Scal As New Mat(5, 1, DepthType.Cv32F, 1)
        Dim Scal2 As New VectorOfDouble({0, 0, 0, 0, 0})

        Dim Test3D1 As New VectorOfPoint3D32F({New MCvPoint3D32f(0.0, 0.0, 0.0),
                                              New MCvPoint3D32f(1.2, 0.0, 0.0),
                                              New MCvPoint3D32f(1.2, 1.8, 0.0),
                                              New MCvPoint3D32f(0.0, 1.8, 0.0)})
        Dim Test3D2 As Matrix(Of Double) = np.ToMatrix({{0.0, 0.0, 0.0},
                                                       {1.2, 0.0, 0.0},
                                                       {1.2, 1.8, 0.0},
                                                       {0.0, 1.8, 0.0}})
        ' Overloaded SolvePNP -> MCvPoint3D32f()
        Dim Test3D3(3) As MCvPoint3D32f
        Test3D3(0) = New MCvPoint3D32f(0.0, 0.0, 0.0)
        Test3D3(1) = New MCvPoint3D32f(1.2, 0.0, 0.0)
        Test3D3(2) = New MCvPoint3D32f(1.2, 1.8, 0.0)
        Test3D3(3) = New MCvPoint3D32f(0.0, 1.8, 0.0)


        Dim Test2D1 As New VectorOfVectorOfPointF({New VectorOfPointF({New PointF(-0.6, -0.9)}),
                                                   New VectorOfPointF({New PointF(0.6, -0.9)}),
                                                   New VectorOfPointF({New PointF(0.6, 0.9)}),
                                                   New VectorOfPointF({New PointF(-0.6, 0.9)})
                                                 })
        Dim Test2D2 As New VectorOfPointF({New PointF(-0.6, -0.9),
                                           New PointF(0.6, -0.9),
                                           New PointF(0.6, 0.9),
                                           New PointF(-0.6, 0.9)
                                         })
        ' Overloaded SolvePNP -> PointF()
        Dim Test2D3(3) As Drawing.PointF
        Test2D3(0) = New PointF(-0.6, -0.9)
        Test2D3(1) = New PointF(0.6, -0.9)
        Test2D3(2) = New PointF(0.6, 0.9)
        Test2D3(3) = New PointF(-0.6, 0.9)

        'Test2D4.Add(New Matrix(Of Double)({1, 2}))
        'corners = np.array (           [
        '                               [[-0.6, -0.9]], 
        '                               [[ 0.6, -0.9]], 
        '                               [[ 0.6,  0.9]], 
        '                               [[-0.6,  0.9]]
        '                              ])


        'array(                      [
        '                            [1.2, 0.0, 0.0 ], 
        '                            [0.0, 1.2, 0.0 ], 
        '                            [0.0, 0.0, 1.0 ]
        '                            ]
        Dim TestKD1 As New VectorOfPoint3D32F({New MCvPoint3D32f(1.2, 0.0, 0.0),
                                              New MCvPoint3D32f(0.0, 1.2, 0.0),
                                              New MCvPoint3D32f(0.0, 0.0, 1.0)})
        Dim TestKD2 As Matrix(Of Double) = np.ToMatrix({{1.2, 0.0, 0.0},
                                                       {0.0, 1.2, 0.0},
                                                       {0.0, 0.0, 1.0}})

        'Emgu.CV.CvInvoke.SolvePnP(Test3D1, Test2D1, TestKD2, Scal, RVec, TVec)

        Emgu.CV.CvInvoke.SolvePnP(Test3D3, Test2D3, TestKD2, Scal, RVec, TVec)

        Dim rvec2 = {BitConverter.ToDouble(RVec.GetData(0, 0), 0), BitConverter.ToDouble(RVec.GetData(0, 1), 0), BitConverter.ToDouble(RVec.GetData(0, 2), 0)}
        Dim tvec2 = {BitConverter.ToDouble(TVec.GetData(0, 0), 0), BitConverter.ToDouble(TVec.GetData(0, 1), 0), BitConverter.ToDouble(TVec.GetData(0, 2), 0)}

        'rvec: array([[ 1.41447611e-11], [-9.48072527e-12], [ 2.69760786e-15]])
        'tvec: array([[-0.6       ], [-0.9       ], [ 1.20000005]])

        main()
    End Sub
End Class

Module page_dewarp

#Region "Parameters"
    Public PAGE_MARGIN_X As Object = 50
    Public PAGE_MARGIN_Y As Object = 20
    Public OUTPUT_ZOOM As Object = 1.0
    Public OUTPUT_DPI As Object = 300
    Public REMAP_DECIMATE As Object = 16
    Public ADAPTIVE_WINSZ As Object = 55
    Public TEXT_MIN_WIDTH As Object = 15
    Public TEXT_MIN_HEIGHT As Object = 2
    Public TEXT_MIN_ASPECT As Object = 1.5
    Public TEXT_MAX_THICKNESS As Object = 10
    Public EDGE_MAX_OVERLAP As Object = 1.0
    Public EDGE_MAX_LENGTH As Object = 100.0
    Public EDGE_ANGLE_COST As Object = 10.0
    Public EDGE_MAX_ANGLE As Object = 7.5
    Public RVEC_IDX As Object = {0, 1, 2, 3} ' slice(0, 3)
    Public TVEC_IDX As Object = {3, 4, 5, 6} ' slice(3, 6)
    Public CUBIC_IDX As Object = {6, 7, 8}   ' slice(6, 8)
    Public SPAN_MIN_WIDTH As Object = 30
    Public SPAN_PX_PER_STEP As Object = 20
    Public FOCAL_LENGTH As Object = 1.2
    Public DEBUG_LEVEL As Object = 0
    Public DEBUG_OUTPUT As Object = "file"
    Public WINDOW_NAME As Object = "Dewarp"
    Public K As Double(,) = {{FOCAL_LENGTH, 0, 0}, {0, FOCAL_LENGTH, 0}, {0, 0, 1}}
    Public CCOLORS As List(Of Double()) = {
                New Double() {255, 0, 0, 0},
                New Double() {255, 63, 0, 0},
                New Double() {255, 127, 0, 0},
                New Double() {255, 191, 0, 0},
                New Double() {255, 255, 0, 0},
                New Double() {191, 255, 0, 0},
                New Double() {127, 255, 0, 0},
                New Double() {63, 255, 0, 0},
                New Double() {0, 255, 0, 0},
                New Double() {0, 255, 63, 0},
                New Double() {0, 255, 127, 0},
                New Double() {0, 255, 191, 0},
                New Double() {0, 255, 255, 0},
                New Double() {0, 191, 255, 0},
                New Double() {0, 127, 255, 0},
                New Double() {0, 63, 255, 0},
                New Double() {0, 0, 255, 0},
                New Double() {63, 0, 255, 0},
                New Double() {127, 0, 255, 0},
                New Double() {191, 0, 255, 0},
                New Double() {255, 0, 255, 0},
                New Double() {255, 0, 191, 0},
                New Double() {255, 0, 127, 0},
                New Double() {255, 0, 63, 0}
 }.ToList
#End Region

    Sub debug_show(ByVal name As Object, ByVal [step] As Object, ByVal text As Object, ByVal display As Object)
        If DEBUG_OUTPUT <> "screen" Then
            Dim filetext = text.replace(" ", "_")
            Dim outfile = name & "_debug_" & Str([step]) & "_" + filetext & ".png"
            Emgu.CV.CvInvoke.Imwrite(outfile, display)
        End If

        If DEBUG_OUTPUT <> "file" Then
            Dim image = display.copy()
            Dim height = image.shape(0)
            Emgu.CV.CvInvoke.PutText(image, text, New Drawing.Point(16, height - 16), FontFace.HersheySimplex, 1.0, New MCvScalar(0, 0, 0), 3, LineType.AntiAlias)
            Emgu.CV.CvInvoke.PutText(image, text, New Drawing.Point(16, height - 16), FontFace.HersheySimplex, 1.0, New MCvScalar(255, 255, 255), 1, LineType.AntiAlias)
            Emgu.CV.CvInvoke.Imshow(WINDOW_NAME, image)
            While Emgu.CV.CvInvoke.WaitKey(5) < 0
            End While
        End If
    End Sub
    Function round_nearest_multiple(ByVal i As Object, ByVal factor As Object) As Object
        i = Convert.ToInt32(i)
        Dim [rem] = i Mod factor

        If Not [rem] Then
            Return i
        Else
            Return i + factor - [rem]
        End If
    End Function

    Function pix2norm(ByVal shape As System.Drawing.Size, pts As Drawing.PointF())
        Dim height = shape.Height
        Dim width = shape.Width
        Dim scl = 2.0 / Math.Max(height, width)
        Dim offset = {width * 0.5, height * 0.5}
        For i As Integer = 0 To pts.Count - 1
            pts(i).X = (pts(i).X - offset(0)) * scl
            pts(i).Y = (pts(i).Y - offset(1)) * scl
        Next
        Return pts
    End Function
    Function pix2norm(ByVal shape As System.Drawing.Size, ByVal pts As List(Of Double())) As List(Of Double())
        Dim height = shape.Height
        Dim width = shape.Width
        Dim scl = 2.0 / Math.Max(height, width)
        Dim offset = {width * 0.5, height * 0.5}
        For Each Entry In pts
            Entry(0) = (Entry(0) - offset(0)) * scl
            Entry(1) = (Entry(1) - offset(1)) * scl
        Next
        Return pts
    End Function

    Function norm2pix(ByVal shape As System.Drawing.Size, ByVal pts As List(Of Double()), ByVal as_integer As Boolean) As Integer
        Dim height = shape.Height
        Dim width = shape.Width
        Dim scl = Math.Max(height, width) * 0.5
        'Dim offset = { 0.5 * width, 0.5 * height }, pts.dtype).reshape(Tuple.Create(-1, 1, 2))
        'Dim rval = pts * scl + offset

        If as_integer Then
            'Return CInt((rval + 0.5))
        Else
            'Return rval
        End If
    End Function

    Function fltp(ByVal point As Double()) As System.Drawing.Point
        Return New System.Drawing.Point(point(0), point(1))
    End Function

    Function draw_correspondences(ByVal img As Object, ByVal dstpoints As Object, ByVal projpts As Object) As Object
        Dim display = img.copy()
        dstpoints = norm2pix(img.shape, dstpoints, True)
        projpts = norm2pix(img.shape, projpts, True)

        For Each _tup_1 In New List(Of Object) From {
            Tuple.Create(projpts, Tuple.Create(255, 0, 0)),
            Tuple.Create(dstpoints, Tuple.Create(0, 0, 255))
        }
            Dim pts = _tup_1.Item1
            Dim color = _tup_1.Item2

            For Each point In pts
                CvInvoke.Circle(display, fltp(point), 3, color, -1, LineType.AntiAlias)
            Next
        Next

        'For Each _tup_2 In zip(projpts, dstpoints)
        '    Dim point_a = _tup_2.Item1
        '    Dim point_b = _tup_2.Item2
        '    cv2.line(display, fltp(point_a), fltp(point_b), Tuple.Create(255, 255, 255), 1, cv2.LINE_AA)
        'Next

        Dim [function] As String = "Test"
        Dim x As Tuple(Of String, Integer, Object)

        Return display
    End Function

    Function get_default_params(ByVal corners As Tuple(Of Double(), Double(), Double(), Double()),
                                ByVal ycoords As List(Of Double()),
                                ByVal xcoords As List(Of Double())) As Object
        Dim page_width As Double = np.linalg.norm(np.Substract(corners.Item2, corners.Item1))
        Dim page_height As Double = np.linalg.norm(np.Substract(corners.Item4, corners.Item1))
        Dim rough_dims = Tuple.Create(page_width, page_height)

        Dim cubic_slopes As Double() = {0.0, 0.0}
        Dim corners_object3d = {{0, 0, 0},
                                {page_width, 0, 0},
                                {page_width, page_height, 0},
                                {0, page_height, 0}}

        Dim CObj3D As Matrix(Of Double) = np.ToMatrix(corners_object3d)
        Dim CamMat As Matrix(Of Double) = np.ToMatrix(K)
        Dim RES As New Mat
        Dim R As New Mat

        Dim ObjKD As New VectorOfPoint3D32F
        Dim k1 As MCvPoint3D32f() = {New MCvPoint3D32f(K(0, 0), K(0, 1), K(0, 2))} : ObjKD.Push(k1)
        Dim k2 As MCvPoint3D32f() = {New MCvPoint3D32f(K(1, 0), K(1, 1), K(1, 2))} : ObjKD.Push(k2)
        Dim k3 As MCvPoint3D32f() = {New MCvPoint3D32f(K(2, 0), K(2, 1), K(2, 2))} : ObjKD.Push(k3)

        Dim Obj3D As New VectorOfPoint3D32F ' OK
        Dim C1 As MCvPoint3D32f() = {New MCvPoint3D32f(corners_object3d(0, 0), corners_object3d(0, 1), corners_object3d(0, 2))} : Obj3D.Push(C1)
        Dim C2 As MCvPoint3D32f() = {New MCvPoint3D32f(corners_object3d(1, 0), corners_object3d(1, 1), corners_object3d(1, 2))} : Obj3D.Push(C2)
        Dim C3 As MCvPoint3D32f() = {New MCvPoint3D32f(corners_object3d(2, 0), corners_object3d(2, 1), corners_object3d(2, 2))} : Obj3D.Push(C3)
        Dim C4 As MCvPoint3D32f() = {New MCvPoint3D32f(corners_object3d(3, 0), corners_object3d(3, 1), corners_object3d(3, 2))} : Obj3D.Push(C4)

        Dim Obj2D As New VectorOfPointF
        Dim DP1 As Drawing.PointF() = {New Drawing.PointF(corners.Item1(0), corners.Item1(1))} : Obj2D.Push(DP1)
        Dim DP2 As Drawing.PointF() = {New Drawing.PointF(corners.Item2(0), corners.Item2(1))} : Obj2D.Push(DP2)
        Dim DP3 As Drawing.PointF() = {New Drawing.PointF(corners.Item3(0), corners.Item3(1))} : Obj2D.Push(DP3)
        Dim DP4 As Drawing.PointF() = {New Drawing.PointF(corners.Item4(0), corners.Item4(1))} : Obj2D.Push(DP4)

        Dim RVec As New Mat
        Dim TVec As New Mat

        Dim Scal As New Mat(5, 1, DepthType.Cv32F, 1)
        CvInvoke.SolvePnP(CObj3D, Obj2D, CamMat, Scal, RVec, TVec)
        Dim rvec2 = {BitConverter.ToDouble(RVec.GetData(0, 0), 0), BitConverter.ToDouble(RVec.GetData(0, 1), 0), BitConverter.ToDouble(RVec.GetData(0, 2), 0)}
        Dim tvec2 = {BitConverter.ToDouble(TVec.GetData(0, 0), 0), BitConverter.ToDouble(TVec.GetData(0, 1), 0), BitConverter.ToDouble(TVec.GetData(0, 2), 0)}

        Throw New Exception("Wrong Values!")
        Dim span_counts As New List(Of Integer) : For Each Entry In xcoords : span_counts.Add(Entry.Length) : Next
        Dim _params = np.hstack(Tuple.Create(rvec, tvec, cubic_slopes, ycoords)) ' + tuple.Create(xcoords))
        Return Tuple.Create(rough_dims, span_counts, _params)
    End Function

    Function project_xy(ByVal xy_coords As Object, ByVal pvec As Object) As Object
        Dim _tup_1 = tuple.Create(pvec(CUBIC_IDX))
        Dim alpha = _tup_1.Item1(0)
        Dim beta = _tup_1.Item1(1)
        Dim poly = np.array(New List(Of Object) From {
            alpha + beta,
            -2 * alpha - beta,
            alpha,
            0
        })
        xy_coords = xy_coords.reshape(Tuple.Create(-1, 2))
        Dim z_coords = np.polyval(poly, xy_coords(":", 0))
        Dim objpoints = np.hstack(Tuple.Create(xy_coords, z_coords.reshape(Tuple.Create(-1, 1))))
        Dim _tup_2 = CvInvoke.ProjectPoints(objpoints, pvec(RVEC_IDX), pvec(TVEC_IDX), np.ToMatrix(K), np.zeros(5))
        Dim image_points = _tup_2.ToArray
        Return image_points
    End Function

    Function project_keypoints(ByVal pvec As Object, ByVal keypoint_index As Object) As Object
        Dim xy_coords = pvec(keypoint_index)
        xy_coords(0, ":") = 0
        Return project_xy(xy_coords, pvec)
    End Function

    Function resize_to_screen(ByVal src As Mat, Optional ByVal maxw As Object = 1280, Optional ByVal maxh As Object = 700, Optional ByVal copy As Object = False) As Mat
        Dim scl_x = CDbl(src.Width) / maxw
        Dim scl_y = CDbl(src.Height) / maxh
        Dim scl = Convert.ToInt32(Math.Max(scl_x, scl_y))

        If scl > 1.0 Then
            Dim ret As New Mat
            Dim inv_scl = 1.0 / scl
            CvInvoke.Resize(src, ret, New Drawing.Size(0, 0), inv_scl, inv_scl, Inter.Area)
            Return ret
        ElseIf copy Then
            Return src.Clone
        Else
            Return src
        End If

        Return src
    End Function

    Function box(ByVal width As Object, ByVal height As Object) As Object
        Return np.ones(Tuple.Create(height, width), dtype:="uint8")
    End Function

    Function get_page_extents(ByVal small As Mat) As Tuple(Of Mat, Integer(,))
        ' -------------------------------------------------
        ' Generiert den Rahmen für das Dokument as neues Mat
        ' -------------------------------------------------
        Dim height As Integer = small.Height
        Dim width As Integer = small.Width
        Dim xmin As Integer = PAGE_MARGIN_X
        Dim ymin As Integer = PAGE_MARGIN_Y
        Dim xmax As Integer = width - PAGE_MARGIN_X
        Dim ymax As Integer = height - PAGE_MARGIN_Y
        Dim page = Mat.Zeros(height, width, DepthType.Cv8U, 1)
        Dim R1 As New System.Drawing.Rectangle(xmin, ymin, xmax - xmin, ymax - ymin)
        Dim C1 As New MCvScalar(255, 255, 255) ' Color
        CvInvoke.Rectangle(page, R1, C1, -1)
        Dim outline = {{xmin, ymin}, {xmin, ymax}, {xmax, ymax}, {xmax, ymin}}
        Return Tuple.Create(Of Mat, Integer(,))(page, outline)
    End Function

    Function get_mask(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal masktype As Object) As Mat
        Dim mask As New Mat
        Dim sgray As New Mat : CvInvoke.CvtColor(small, sgray, ColorConversion.Rgb2Gray)

        If masktype = "text" Then
            CvInvoke.AdaptiveThreshold(sgray, mask, 255, AdaptiveThresholdType.MeanC, ThresholdType.BinaryInv, ADAPTIVE_WINSZ, 25)
            Dim Box91 As Emgu.CV.Mat = Emgu.CV.CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, New System.Drawing.Size(9, 1), New Drawing.Point(-1, -1)) ' 9,1
            Dim Box13 As Emgu.CV.Mat = Emgu.CV.CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, New System.Drawing.Size(1, 3), New Drawing.Point(-1, -1)) ' 1,3
            Emgu.CV.CvInvoke.Dilate(mask, mask, Box91, New System.Drawing.Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, Nothing)
            Emgu.CV.CvInvoke.Erode(mask, mask, Box13, New System.Drawing.Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, Nothing)
        Else
            CvInvoke.AdaptiveThreshold(sgray, sgray, 255, AdaptiveThresholdType.MeanC, ThresholdType.BinaryInv, ADAPTIVE_WINSZ, 7)
            Dim Box31 As Emgu.CV.Mat = Emgu.CV.CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, New System.Drawing.Size(3, 1), New Drawing.Point(-1, -1)) ' 3,1
            Dim Box82 As Emgu.CV.Mat = Emgu.CV.CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, New System.Drawing.Size(8, 2), New Drawing.Point(-1, -1)) ' 8,2
            Emgu.CV.CvInvoke.Erode(mask, mask, Box31, New System.Drawing.Point(-1, -1), 3, Emgu.CV.CvEnum.BorderType.Default, Nothing)
            Emgu.CV.CvInvoke.Dilate(mask, mask, Box82, New System.Drawing.Point(-1, -1), 1, Emgu.CV.CvEnum.BorderType.Default, Nothing)
        End If

        Return np.minimum(mask, pagemask)
    End Function

    Function interval_measure_overlap(ByVal int_a As Tuple(Of Double, Double), ByVal int_b As Tuple(Of Double, Double)) As Double
        Return Math.Min(int_a.Item2, int_b.Item2) - Math.Max(int_a.Item1, int_b.Item1)
    End Function

    Function angle_dist(ByVal angle_b As Object, ByVal angle_a As Object) As Object
        Dim diff = angle_b - angle_a

        While diff > Math.PI
            diff -= 2 * Math.PI
        End While

        While diff < -Math.PI
            diff += 2 * Math.PI
        End While

        Return Math.Abs(diff)
    End Function

    Function blob_mean_and_tangent(ByVal contour As VectorOfPoint) As Object
        Dim moments = CvInvoke.Moments(contour)
        Dim area = moments.M00
        Dim mean_x = moments.M10 / area
        Dim mean_y = moments.M01 / area
        Dim moments_matrix = {{moments.Mu20, moments.Mu11}, {moments.Mu11, moments.Mu02}}
        Dim Mom As New Matrix(Of Double)(2, 2)
        Mom.Data(0, 0) = moments.Mu20
        Mom.Data(0, 1) = moments.Mu11
        Mom.Data(1, 0) = moments.Mu11
        Mom.Data(1, 1) = moments.Mu02
        Dim svd_u As New Mat : CvInvoke.SVDecomp(Mom, New Mat, svd_u, New Mat, CvEnum.SvdFlag.Default)
        Dim center = {mean_x, mean_y} ' OK

        ' tangent = svd_u[:, 0].flatten().copy()
        Dim tangent As Double() = {BitConverter.ToDouble(svd_u.GetData, 0), BitConverter.ToDouble(svd_u.GetData, 16)} ' OK

        Return Tuple.Create(center, tangent)
    End Function

    Function generate_candidate_edge(ByVal cinfo_a As ContourInfo, ByVal cinfo_b As ContourInfo) As Object
        If cinfo_a.point0(0) > cinfo_b.point1(0) Then
            Dim tmp = cinfo_a
            cinfo_a = cinfo_b
            cinfo_b = tmp
        End If

        Dim x_overlap_a = cinfo_a.local_overlap(cinfo_b)
        Dim x_overlap_b = cinfo_b.local_overlap(cinfo_a)
        Dim overall_tangent As Double() = np.Substract(cinfo_b.center, cinfo_a.center) ' cinfo_b.center - cinfo_a.center
        Dim overall_angle = np.arctan2(overall_tangent(1), overall_tangent(0))
        Dim delta_angle = Math.Max(angle_dist(cinfo_a.angle, overall_angle), angle_dist(cinfo_b.angle, overall_angle)) * 180 / Math.PI
        Dim x_overlap = Math.Max(x_overlap_a, x_overlap_b)

        Dim dist = np.linalg.norm(np.Substract(cinfo_b.point0, cinfo_a.point1))

        If dist > EDGE_MAX_LENGTH OrElse x_overlap > EDGE_MAX_OVERLAP OrElse delta_angle > EDGE_MAX_ANGLE Then
            Return Nothing
        Else
            Dim score = dist + delta_angle * EDGE_ANGLE_COST
            Return Tuple.Create(score, cinfo_a, cinfo_b)
        End If
    End Function

    Function make_tight_mask(ByVal contour As VectorOfPoint, ByVal xmin As Object, ByVal ymin As Object, ByVal width As Object, ByVal height As Object) As Object
        Dim tight_mask As Mat = np.zeros(Tuple.Create(height, width), DepthType.Cv8U, 1) ' OK
        Dim Color As New MCvScalar(1, 1, 1)

        Dim C1 As New VectorOfPoint
        Dim C2 As New Matrix(Of Double)({{xmin, ymin}})  ' np.array(Tuple.Create(xmin, ymin)).reshape(Tuple.Create(-1, 1, 2))
        Emgu.CV.CvInvoke.Subtract(contour, C2, C1) ' contour - [C2]
        Dim C3 As New VectorOfVectorOfPoint : C3.Push(C1)

        ' -------- just for debug to show the mask structure ---------
        'Dim tight_mask_debug As Mat = np.zeros(Tuple.Create(height, width), DepthType.Cv8U, 1) ' OK
        'Dim Color_debug As New MCvScalar(255,255, 255) ' Need other color to view the mask
        'CvInvoke.DrawContours(tight_mask_debug, C3, 0, Color_debug, -1)
        ' ------------------------------------------------------------

        CvInvoke.DrawContours(tight_mask, C3, 0, Color, -1)
        Return tight_mask
    End Function

    Function get_contours(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal masktype As Object) As Object
        Dim mask As Mat = get_mask(name, small, pagemask, masktype) ' OK
        Dim contours As New VectorOfVectorOfPoint()                 ' OK

        CvInvoke.FindContours(mask,
                              contours,
                              New Mat,
                              RetrType.External,
                              ChainApproxMethod.ChainApproxNone)    ' OK
        Dim contours_out = New List(Of ContourInfo)()

        For i As Integer = 0 To contours.Size - 1                   ' OK 
            Dim rect = CvInvoke.BoundingRectangle(contours(i))      ' OK

            If rect.Width < TEXT_MIN_WIDTH OrElse
                rect.Height < TEXT_MIN_HEIGHT OrElse
                rect.Width < TEXT_MIN_ASPECT * rect.Height Then Continue For ' OK

            Dim tight_mask As Mat = make_tight_mask(contours(i), rect.X, rect.Y, rect.Width, rect.Height)

            ' ------------------------------------------------------------------------
            ' If tight_mask.sum(axis:=0).max() > TEXT_MAX_THICKNESS Then Continue For
            ' Nach besserer Lösung suchen ... 
            Dim Test_Sum = SumAxis0Max(tight_mask.GetData, tight_mask.Width, tight_mask.Height)
            If Test_Sum > TEXT_MAX_THICKNESS Then Continue For
            ' ------------------------------------------------------------------------

            contours_out.Add(New ContourInfo(contours(i), rect, tight_mask))
        Next

        visualize_contours(name, small, contours_out)

        Return contours_out
    End Function

    Function assemble_spans(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal cinfo_list As List(Of ContourInfo)) As Object

        cinfo_list = cinfo_list.OrderBy(Function(x) x.rect.y).ToList ' cinfo_list = cinfo_list.OrderBy(Function(cinfo) cinfo.rect(0)).ToList()
        Dim candidate_edges = New List(Of Object)()

        For i As Integer = 0 To cinfo_list.Count - 1
            Dim cinfo_i = cinfo_list(i)
            For j = 0 To i : If j = i Then Continue For
                Dim edge = generate_candidate_edge(cinfo_i, cinfo_list(j))
                If edge IsNot Nothing Then candidate_edges.Add(edge)
            Next
        Next

        candidate_edges.Sort()

        For Each _tup_2 In candidate_edges
            Dim cinfo_a = _tup_2.Item2
            Dim cinfo_b = _tup_2.Item3

            If cinfo_a.succ Is Nothing AndAlso cinfo_b.pred Is Nothing Then
                cinfo_a.succ = cinfo_b
                cinfo_b.pred = cinfo_a
            End If
        Next

        Dim spans = New List(Of ContourInfo())

        While cinfo_list.Count > 0
            Dim cinfo = cinfo_list(0)

            While IsNothing(cinfo.pred) = False
                cinfo = cinfo.pred
            End While

            Dim cur_span = New List(Of ContourInfo)
            Dim width = 0.0

            While Not IsNothing(cinfo)
                cinfo_list.Remove(cinfo)
                cur_span.Add(cinfo)
                width += cinfo.local_xrng.Item2 - cinfo.local_xrng.Item1
                cinfo = cinfo.succ
            End While


            If width > SPAN_MIN_WIDTH Then
                spans.Add(cur_span.ToArray)
            End If
        End While

        visualize_spans(name, small, pagemask, spans)

        Return spans
    End Function

    Function sample_spans(ByVal shape As Object, ByVal spans As Object) As List(Of List(Of Double()))
        Dim span_points = New List(Of List(Of Double()))()

        For Each span In spans
            Dim contour_points = New List(Of Double())

            For Each cinfo In span
                Dim MatArray As Byte(,) = np.MatToInteger(cinfo.mask)
                Dim Totals2 As Byte(,) = np.mulLine(MatArray) ' macht keinen sinn bei 1.Zeile (* 0 = 0) ...
                Dim MaskSum As Integer() = np.sum(Totals2, 0)
                Dim Means1 As Integer() = np.sum(MatArray, 0)
                Dim Means As Double() = np.div(MaskSum, Means1)

                Dim xmin = cinfo.rect.x
                Dim ymin = cinfo.rect.y
                Dim [step] = SPAN_PX_PER_STEP
                Dim start = CInt(Math.Floor(((Means.Length - 1) Mod [step]) / 2)) ' 3 / 2 ist in Python 1 (abschneiden von dezimal = floor, dann Cint)
                'contour_points += Range(start, Means.Count, [step]).[Select](Function(x) Tuple.Create(x + xmin, Means(x) + ymin))
                For i As Integer = start To Means.Count Step [step]
                    contour_points.Add({i + xmin, Means(i) + ymin})
                Next
            Next

            'contour_points = np.array(contour_points, dtype:="double").reshape(Tuple.Create(-1, 1, 2))
            contour_points = pix2norm(shape, contour_points)
            span_points.Add(contour_points)

        Next

        Return span_points
    End Function

    Function keypoints_from_samples(ByVal name As String,
                                    ByVal small As Mat,
                                    ByVal pagemask As Mat,
                                    ByVal page_outline As Object,
                                    ByVal span_points As List(Of List(Of Double()))) As Object

        Dim all_evecs As Double() = {0.0, 0.0}
        Dim Evec As New VectorOfFloat
        Dim all_weights As Double = 0

        For Each points In span_points
            Dim Matrix As New Matrix(Of Double)(points.Count, 2)
            For i As Integer = 0 To CInt(points.Count) - 1
                Dim NewMat As New Matrix(Of Double)(2, 2)
                Dim newSca As New MCvScalar(points(i)(0), points(i)(0))
                Matrix(i, 0) = points(i)(0)
                Matrix(i, 1) = points(i)(1)
            Next
            CvInvoke.PCACompute(Matrix, New Mat, Evec, maxComponents:=1)

            Dim weight = np.linalg.norm(np.Substract(points(points.Count - 1), points(0)))
            all_evecs = (np.add(all_evecs, np.mul(Evec.ToArray, weight)))
            all_weights += weight
        Next

        Evec = New VectorOfFloat(np.div(all_evecs, all_weights))
        Dim x_dir = Evec '  np.Flatten(Evec)

        If x_dir(0) < 0 Then
            'x_dir(0) = x_dir(0) * -1
            Evec = New VectorOfFloat({Evec(0) * -1, Evec(1) * -1})
        End If

        Dim y_dir = {x_dir(1) * -1, x_dir(0)}

        Dim Outline(3) As Drawing.PointF
        Outline(0) = New Drawing.PointF(page_outline(0, 0), page_outline(0, 1))
        Outline(1) = New Drawing.PointF(page_outline(1, 0), page_outline(1, 1))
        Outline(2) = New Drawing.PointF(page_outline(2, 0), page_outline(2, 1))
        Outline(3) = New Drawing.PointF(page_outline(3, 0), page_outline(3, 1))

        Dim pagecoords = CvInvoke.ConvexHull(Outline)
        pagecoords = pix2norm(pagemask.Size, pagecoords)
        Dim px_coords As Double() = np.dot(pagecoords, x_dir.ToArray)
        Dim py_coords As Double() = np.dot(pagecoords, y_dir.ToArray)
        Dim px0 = px_coords.Min
        Dim px1 = px_coords.Max
        Dim py0 = py_coords.Min
        Dim py1 = py_coords.Max
        Dim p00 = np.add(np.mul(x_dir.ToArray, px0), np.mul(y_dir, py0))
        Dim p10 = np.add(np.mul(x_dir.ToArray, px1), np.mul(y_dir, py0))
        Dim p11 = np.add(np.mul(x_dir.ToArray, px1), np.mul(y_dir, py1))
        Dim p01 = np.add(np.mul(x_dir.ToArray, px0), np.mul(y_dir, py1))
        Dim corners = np.vstack(Tuple.Create(p00, p10, p11, p01))

        Dim ycoords As List(Of Double()) = New List(Of Double())
        Dim xcoords As List(Of Double()) = New List(Of Double())
        For Each points In span_points
            Dim pts As List(Of Double()) = points
            px_coords = np.dot(pts, x_dir.ToArray)
            py_coords = np.dot(pts, y_dir.ToArray)
            ycoords.Add({np.Mean(py_coords) - py0})
            xcoords.Add(np.Substract(px_coords, px0))
        Next

        'If DEBUG_LEVEL >= 2 Then
        visualize_span_points(name, small, span_points, corners)
        'End If

        Return Tuple.Create(corners, ycoords, xcoords)
    End Function

    Function visualize_contours(ByVal name As String, ByVal small As Mat, ByVal cinfo_list As List(Of ContourInfo)) As Object
        Dim cinfo As Object
        Dim regions = np.zeros_like(small)

        For j As Integer = 0 To cinfo_list.Count - 1
            cinfo = cinfo_list(j)
            Dim Color As Double() = CCOLORS(j Mod CCOLORS.Count)
            Dim SpanArr As New VectorOfVectorOfPoint : SpanArr.Push(cinfo.contour)
            CvInvoke.DrawContours(regions, SpanArr, 0, New MCvScalar(Color(0), Color(1), Color(1)), -1)
        Next

        Dim Display As Mat = small.Clone
        'Dim mask = regions.max(axis:=2) <> 0
        'Dim display = small.Clone()
        'display(mask) = display(mask) / 2 + regions(mask) / 2

        For j As Integer = 0 To cinfo_list.Count - 1
            cinfo = cinfo_list(j)
            Dim Color = CCOLORS(j * 4 Mod CCOLORS.Count)
            Dim Col As New MCvScalar(Color(0) / 4, Color(1) / 4, Color(2) / 4)
            CvInvoke.Circle(Display, New Drawing.Point(cinfo.center(0), cinfo.center(1)), 3, Col, 1, LineType.AntiAlias)
            CvInvoke.Line(Display, New Drawing.Point(cinfo.point0(0), cinfo.point0(1)),
                                    New Drawing.Point(cinfo.point1(0), cinfo.point1(1)), Col, 1, LineType.AntiAlias)
        Next

        'For Each _tup_2 In cinfo_list.[Select](Function(_p_3, _p_4) Tuple.Create(_p_4, _p_3))
        '    Dim Col As New MCvScalar(255, 255, 255)
        '    j = _tup_2.Item1
        '    cinfo = _tup_2.Item2
        '    Dim color = CCOLORS(j Mod CCOLORS.Count)
        '    color = tuple.Create(color.[Select](Function(c) c / 4))
        '    CvInvoke.Circle(display, fltp(cinfo.center), 3, Col, 1, LineType.AntiAlias)
        '    CvInvoke.Line(display, fltp(cinfo.point0), fltp(cinfo.point1), Col, 1, LineType.AntiAlias)
        'Next

        debug_show(name, 1, "contours", Display)
    End Function

    Function visualize_spans(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal spans As List(Of ContourInfo())) As Object
        Dim regions = np.zeros_like(small)

        For i As Integer = 0 To spans.Count - 1
            For Each Span In spans(i)
                If Span.contour.Size = 0 Then Continue For
                Dim SpanArr As New VectorOfVectorOfPoint : SpanArr.Push(Span.contour)
                Dim Color As Double() = CCOLORS(i * 4 Mod CCOLORS.Count)
                Dim ColorMCV As New MCvScalar(Color(0), Color(1), Color(2))
                CvInvoke.DrawContours(regions, SpanArr, -1, ColorMCV, 1)
            Next
        Next

        'Dim mask = regions.max(axis:=2) <> 0
        'Dim display = small.copy()
        'display(mask) = display(mask) / 2 + regions(mask) / 2
        ' display[pagemask == 0] /= 4;
        ' debug_show(name, 2, "spans", display)
    End Function

    Function visualize_span_points(ByVal name As Object, ByVal small As Mat, ByVal span_points As List(Of List(Of Double())), ByVal corners As Object) As Object
        Dim display = small
        Dim Col As New MCvScalar(255, 255, 255)

        ' For Each _tup_1 In span_points.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
        For iSpan As Integer = 0 To span_points.Count
            ' Dim points = span_points(iSpan)
            'points = norm2pix(small.Size, points, False)
            'Dim Tup As New MCvScalar(-1, 2)
            'Dim _tup_2 = CvInvoke.PCACompute(points, _tup_2, Nothing, maxComponents:=1)
            'Dim mean = _tup_2.Item1
            'Dim small_evec = _tup_2.Item2
            Dim dps = Nothing ' np.dot(points.reshape(Tuple.Create(-1, 2)), small_evec.reshape(Tuple.Create(2, 1)))
            'Dim dpm = np.dot(np.Flatten(mean), np.Flatten(small_evec))
            'Dim point0 = mean + small_evec * (dps.min() - dpm)
            'Dim point1 = mean + small_evec * (dps.max() - dpm)

            'For Each point In points
            ' CvInvoke.Circle(display, fltp(point), 3, CCOLORS(CDbl(i Mod CCOLORS.Count)), -1, LineType.AntiAlias)
            'Next

            'CvInvoke.Line(display, fltp(point0), fltp(point1), Col, 1, LineType.AntiAlias)
        Next

        'CvInvoke.Polylines(display, New List(Of Object) From {norm2pix(small.Size, corners, True)}, True, Col)
        debug_show(name, 3, "span points", display)
    End Function

    Function imgsize(ByVal img As Object) As Object
        Dim _tup_1 = img.shape(2)
        Dim height = _tup_1.Item1
        Dim width = _tup_1.Item2
        Return "{}x{}".Format(width, height)
    End Function

    Function make_keypoint_index(ByVal span_counts As Object) As Object
        Dim nspans = span_counts.Count
        Dim npts = span_counts.Sum()
        Dim keypoint_index = np.zeros(Tuple.Create(npts + 1, 2), "int")
        Dim start = 1

        For Each _tup_1 In span_counts.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
            Dim i = _tup_1.Item1
            Dim count = _tup_1.Item2
            Dim [end] = start + count
            ' keypoint_index(start(start + [end]), 1) = 8 + i
            start = [end]
        Next

        keypoint_index(1, 0) = np.arange(npts) + 8 + nspans
        Return keypoint_index
    End Function

    Function optimize_params(ByVal name As Object, ByVal small As Object, ByVal dstpoints As Object, ByVal span_counts As Object, ByVal _params As Object) As Object
        Dim display As Object
        Dim projpts As Object
        Dim keypoint_index = make_keypoint_index(span_counts)
        Dim objective As Func(Of Object, Object) = Function(pvec)
                                                       Dim ppts = project_keypoints(pvec, keypoint_index)
                                                       'Return np.sum(Math.Pow(dstpoints - ppts, 2))
                                                   End Function

        If DEBUG_LEVEL >= 1 Then
            projpts = project_keypoints(_params, keypoint_index)
            display = draw_correspondences(small, dstpoints, projpts)
            debug_show(name, 4, "keypoints before", display)
        End If

        Dim res = New scipy.optimize().minimize(objective, _params, method:="Powell")
        _params = res.x

        If DEBUG_LEVEL >= 1 Then
            projpts = project_keypoints(_params, keypoint_index)
            display = draw_correspondences(small, dstpoints, projpts)
            debug_show(name, 5, "keypoints after", display)
        End If

        Return _params
    End Function

    Function get_page_dims(ByVal corners As Object, ByVal rough_dims As Object, ByVal _params As Object) As Object
        Dim dst_br = np.Flatten(corners.item3) ' .flatten()
        Dim dims As Object = np.array(rough_dims)
        Dim objective As Func(Of Object, Object) = Function(dims_)
                                                       Dim proj_br = project_xy(dims_, _params)
                                                       ' Return np.sum(Math.Pow(dst_br - proj_br.flatten(), 2))
                                                   End Function

        Dim res = New scipy.optimize().minimize(objective, dims, method:="Powell") ' Todo - Link to Function needed
        dims = res.x
        Return dims
    End Function

    Function remap_image(ByVal name As String, ByVal img As Object, ByVal small As Mat, ByVal page_dims As Mat, ByVal _params As Object) As Object
        Dim height = 0 ' 0.5 * page_dims.GetData(1) * OUTPUT_ZOOM * img.size.height
        height = round_nearest_multiple(height, REMAP_DECIMATE)
        Dim width = 0 ' round_nearest_multiple(height * page_dims.GetData(0) / page_dims.GetData(1), REMAP_DECIMATE)
        Dim height_small = height / REMAP_DECIMATE
        Dim width_small = width / REMAP_DECIMATE
        Dim page_x_range = 0 '  np.linspace(0, page_dims.GetData(0), width_small)
        Dim page_y_range = 0 'np.linspace(0, page_dims.GetData(1), height_small)
        Dim _tup_1 = np.meshgrid(page_x_range, page_y_range)
        Dim page_x_coords = _tup_1.Item1
        Dim page_y_coords = _tup_1.Item2
        Dim page_xy_coords = np.hstack(Tuple.Create(page_x_coords.flatten().reshape(Tuple.Create(-1, 1)), page_y_coords.flatten().reshape(Tuple.Create(-1, 1))))
        page_xy_coords = page_xy_coords.astype(Of Double)()
        Dim image_points = project_xy(page_xy_coords, _params)
        image_points = norm2pix(img.shape, image_points, False)
        Dim image_x_coords = image_points(":", 0, 0).reshape(page_x_coords.shape)
        Dim image_y_coords = image_points(":", 0, 1).reshape(page_y_coords.shape)
        CvInvoke.Resize(image_x_coords, image_x_coords, New Drawing.Size(width, height), interpolation:=Inter.Cubic)
        CvInvoke.Resize(image_y_coords, image_y_coords, New Drawing.Size(width, height), interpolation:=Inter.Cubic)
        Dim img_gray As Mat = Nothing : CvInvoke.CvtColor(img, img_gray, ColorConversion.Rgb2Gray)
        Dim remapped As Mat = Nothing : CvInvoke.Remap(img, remapped, image_x_coords, image_y_coords, Inter.Cubic, BorderType.Replicate, Nothing)
        Dim thresh As Mat = Nothing : CvInvoke.AdaptiveThreshold(remapped, thresh, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, ADAPTIVE_WINSZ, 25)

        'Dim pil_image = Image.fromarray(thresh)
        'pil_image = pil_image.convert("1")
        'Dim threshfile = name & "_thresh.png"
        'pil_image.save(threshfile, dpi:=Tuple.Create(OUTPUT_DPI, OUTPUT_DPI))

        'Return threshfile
    End Function

    Function main() As Object
        Dim outfiles = New List(Of Object)()

        Dim imgfile As String = "Test.jpg"          ' OK cross
        Dim img As Mat = CvInvoke.Imread(imgfile)   ' OK cross
        Dim small As Mat = resize_to_screen(img)    ' OK cross
        Dim name As String = imgfile                ' OK cross

        Dim PageExtents = get_page_extents(small)                               ' OK cross
        Dim cinfo_list = get_contours(name, small, PageExtents.Item1, "text")   ' OK cross
        Dim spans = assemble_spans(name, small, PageExtents.Item1, cinfo_list)

        If spans.Count < 3 Then
            Console.WriteLine("  detecting lines because only", spans.Count, "text spans")
            cinfo_list = get_contours(name, small, PageExtents.Item1, "line")
            Dim spans2 = assemble_spans(name, small, PageExtents.Item1, cinfo_list)
            If spans2.Count > spans.Count Then spans = spans2
        End If

        If spans.Count < 1 Then Return Nothing
        Dim span_points As List(Of List(Of Double())) = sample_spans(small.Size, spans)
        Dim _tup_3 = keypoints_from_samples(name, small, PageExtents.Item1, PageExtents.Item2, span_points)
        ' ------- Werte aus Tuppel --------
        Dim corners = _tup_3.Item1
        Dim ycoords As List(Of Double()) = _tup_3.Item2
        Dim xcoords As List(Of Double()) = _tup_3.Item3

        Dim _tup_4 = get_default_params(corners, ycoords, xcoords)
        ' ------- Werte aus Tuppel --------
        Dim rough_dims = _tup_4.Item1
        Dim span_counts = _tup_4.Item2
        Dim _params = _tup_4.Item3

        'Dim dstpoints = np.vstack(Tuple.Create(corners(0).reshape(Tuple.Create(1, 1, 2))) + tuple.Create(span_points))
        ' Dim params = optimize_params(name, small, dstpoints, span_counts, _params)
        Dim page_dims = get_page_dims(corners, rough_dims, _params)
        Dim outfile = remap_image(name, img, small, page_dims, _params)

    End Function

    Function SumAxis0Max(ByVal Data As Byte(), ByVal width As Integer, ByVal Height As Integer) As Integer
        Dim MaxData As Integer = 0
        Dim NewData As Byte(,) = Accord.Math.Matrix.Reshape(Data, Height, width, Accord.Math.MatrixOrder.Default)

        For i As Integer = 0 To width - 1
            Dim LocalMax As Integer = 0
            Dim ColData As Byte() = Accord.Math.Matrix.GetColumn(NewData, 1)
            For i2 As Integer = 0 To ColData.Count - 1
                LocalMax += ColData(i2)
            Next
            If LocalMax > MaxData Then MaxData = LocalMax
        Next
        Return MaxData
    End Function
    Function Min(ByVal a As List(Of Double))
        Return a.Min()
    End Function
    Function Max(ByVal a As List(Of Double))
        Return a.Max()
    End Function
End Module


Class scipy
    Class optimize
        Function minimize(ByVal a, ByVal b, ByVal method)
            MsgBox(1)
            Return Nothing
        End Function
    End Class
End Class