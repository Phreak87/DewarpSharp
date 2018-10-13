
Imports Emgu.CV
Imports Emgu.CV.Util
Imports Emgu.CV.CvEnum
Imports Emgu.CV.Structure

Imports System.Collections.Generic

Public Class Form1

    Private Sub Form1_Load(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles MyBase.Load

    End Sub

    Private Sub Button1_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles Button1.Click
        main()
    End Sub
End Class

Module page_dewarp
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
    Public K = {{FOCAL_LENGTH, 0, 0}, {0, FOCAL_LENGTH, 0}, {0, 0, 1}}
    Public CCOLORS = {
                {255, 0, 0}, {255, 63, 0}, {255, 127, 0}, {255, 191, 0},
                {255, 255, 0}, {191, 255, 0}, {127, 255, 0}, {63, 255, 0},
                {0, 255, 0}, {0, 255, 63}, {0, 255, 127}, {0, 255, 191},
                {0, 255, 255}, {0, 191, 255}, {0, 127, 255}, {0, 63, 255},
                {0, 0, 255}, {63, 0, 255}, {127, 0, 255}, {191, 0, 255},
                {255, 0, 255}, {255, 0, 191}, {255, 0, 127}, {255, 0, 63}
 }


    Function debug_show(ByVal name As Object, ByVal [step] As Object, ByVal text As Object, ByVal display As Object) As Object
        If DEBUG_OUTPUT <> "screen" Then
            Dim filetext = text.replace(" ", "_")
            Dim outfile = name & "_debug_" & str([step]) & "_" + filetext & ".png"
            'cv2.imwrite(outfile, display)
        End If

        If DEBUG_OUTPUT <> "file" Then
            Dim image = display.copy()
            Dim height = image.shape(0)
            'cv2.putText(image, text, Tuple.Create(16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Tuple.Create(0, 0, 0), 3, cv2.LINE_AA)
            'cv2.putText(image, text, Tuple.Create(16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Tuple.Create(255, 255, 255), 1, cv2.LINE_AA)
            'cv2.imshow(WINDOW_NAME, image)

            'While cv2.waitKey(5) < 0
            'End While
        End If
    End Function

    Function round_nearest_multiple(ByVal i As Object, ByVal factor As Object) As Object
        i = Convert.ToInt32(i)
        Dim [rem] = i Mod factor

        If Not [rem] Then
            Return i
        Else
            Return i + factor - [rem]
        End If
    End Function

    Function pix2norm(ByVal shape As Object, ByVal pts As Object) As Object
        Dim _tup_1 = shape(2)
        Dim height = _tup_1.Item1
        Dim width = _tup_1.Item2
        Dim scl = 2.0 / Math.Max(height, width)
        Dim offset = np.array(New List(Of Object) From {
            width,
            height
        }, dtype:=pts.dtype).reshape(Tuple.Create(-1, 1, 2)) * 0.5
        Return (pts - offset) * scl
    End Function

    Function norm2pix(ByVal shape As Object, ByVal pts As Object, ByVal as_integer As Object) As Object
        'Dim _tup_1 = shape(2)
        'Dim height = _tup_1.Item1
        'Dim width = _tup_1.Item2
        'Dim scl = Math.Max(height, width) * 0.5
        'Dim offset = { 0.5 * width, 0.5 * height }, pts.dtype).reshape(Tuple.Create(-1, 1, 2))
        'Dim rval = pts * scl + offset

        'If as_integer Then
        '    Return CInt((rval + 0.5))
        'Else
        '    Return rval
        'End If
    End Function

    Function fltp(ByVal point As Object) As Object
        ' Return tuple(CInt(point.flatten()))
        Return New Mat(CInt(point.flatten()))
    End Function

    Function draw_correspondences(ByVal img As Object, ByVal dstpoints As Object, ByVal projpts As Object) As Object
        'Dim display = img.copy()
        'dstpoints = norm2pix(img.shape, dstpoints, True)
        'projpts = norm2pix(img.shape, projpts, True)

        'For Each _tup_1 In New List(Of Object) From {
        '    Tuple.Create(projpts, Tuple.Create(255, 0, 0)),
        '    Tuple.Create(dstpoints, Tuple.Create(0, 0, 255))
        '}
        '    Dim pts = _tup_1.Item1
        '    Dim color = _tup_1.Item2

        '    For Each point In pts
        '        CvInvoke.Circle(display, fltp(point), 3, color, -1, LineType.AntiAlias)
        '    Next
        'Next

        'For Each _tup_2 In zip(projpts, dstpoints)
        '    Dim point_a = _tup_2.Item1
        '    Dim point_b = _tup_2.Item2
        '    cv2.line(display, fltp(point_a), fltp(point_b), Tuple.Create(255, 255, 255), 1, cv2.LINE_AA)
        'Next

        'Dim [function] As String = "Test"
        'Dim x As Tuple(Of String, Integer, Object)

        'Return display
    End Function

    Function get_default_params(ByVal corners As Object, ByVal ycoords As Object, ByVal xcoords As Object) As Object
        'Dim page_width = np.linalg.norm(corners(1) - corners(0))
        'Dim page_height = np.linalg.norm(corners(-1) - corners(0))
        'Dim rough_dims = Tuple.Create(page_width, page_height)
        'Dim cubic_slopes = {0.0, 0.0}
        'Dim corners_object3d = {{0, 0, 0}, {page_width, 0, 0}, {page_width, page_height, 0}, {0, page_height, 0}}
        'Dim _tup_1 : CvInvoke.SolvePnP(corners_object3d, corners, K, New MCvScalar(5), Nothing, _tup_1)
        'Dim rvec = _tup_1.Item2
        'Dim tvec = _tup_1.Item3
        'Dim span_counts = xcoords.[Select](Function(xc) xc.Count)
        'Dim _params = np.hstack(Tuple.Create(np.array(rvec).flatten(), np.array(tvec).flatten(), np.array(cubic_slopes).flatten(), ycoords.flatten()) + tuple.Create(xcoords))
        'Return Tuple.Create(rough_dims, span_counts, _params)
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
        Dim _tup_2 = CvInvoke.ProjectPoints(objpoints, pvec(RVEC_IDX), pvec(TVEC_IDX), K, np.zeros(5))
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

    Function get_page_extents(ByVal small As Mat) As Object
        Dim height = small.Height
        Dim width = small.Width
        Dim xmin = PAGE_MARGIN_X
        Dim ymin = PAGE_MARGIN_Y
        Dim xmax = width - PAGE_MARGIN_X
        Dim ymax = height - PAGE_MARGIN_Y
        Dim page = Mat.Zeros(height, width, DepthType.Cv8U, 1)
        Dim R1 As New System.Drawing.Rectangle(xmin, ymin, xmax - xmin, ymax - ymin)
        Dim C1 As New MCvScalar(255, 255, 255) ' Color
        CvInvoke.Rectangle(page, R1, C1, -1)
        Dim outline = {{xmin, ymin}, {xmin, ymax}, {xmax, ymax}, {xmax, ymin}}
        Return Tuple.Create(page, outline)
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

    Function interval_measure_overlap(ByVal int_a As Object, ByVal int_b As Object) As Object
        Return Math.Min(int_a(1), int_b(1)) - Math.Max(int_a(0), int_b(0))
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
        Dim center = {mean_x, mean_y}
        Dim tangent = svd_u.GetData ' svd_u(":",0).flatten().copy() ???? 
        Return Tuple.Create(center, tangent)
    End Function

    Public Class ContourInfo
        Property contour
        Property rect
        Property mask
        Property center
        Property tangent
        Property angle
        Property local_xrng
        Property point0
        Property point1
        Property pred
        Property succ

        Public Sub New(ByVal contour As Object, ByVal rect As Object, ByVal mask As Object)
            Me.contour = contour
            Me.rect = rect
            Me.mask = mask
            Dim _tup_1 = blob_mean_and_tangent(contour)
            Me.center = _tup_1.Item1
            Me.tangent = _tup_1.Item2
            Me.angle = np.arctan2(Me.tangent(1), Me.tangent(0))
            Dim clx = contour.[Select](Function(point) Me.proj_x(point))
            Dim lxmin = Min(clx)
            Dim lxmax = max(clx)
            Me.local_xrng = Tuple.Create(lxmin, lxmax)
            Me.point0 = Me.center + Me.tangent * lxmin
            Me.point1 = Me.center + Me.tangent * lxmax
            Me.pred = Nothing
            Me.succ = Nothing
        End Sub

        Public Overridable Function proj_x(ByVal point As Object) As Object
            Return np.dot(Me.tangent, point.flatten() - Me.center)
        End Function

        Public Overridable Function local_overlap(ByVal other As Object) As Object
            Dim xmin = Me.proj_x(other.point0)
            Dim xmax = Me.proj_x(other.point1)
            Return interval_measure_overlap(Me.local_xrng, Tuple.Create(xmin, xmax))
        End Function
    End Class

    Function generate_candidate_edge(ByVal cinfo_a As Object, ByVal cinfo_b As Object) As Object
        If cinfo_a.point0(0) > cinfo_b.point1(0) Then
            Dim tmp = cinfo_a
            cinfo_a = cinfo_b
            cinfo_b = tmp
        End If

        Dim x_overlap_a = cinfo_a.local_overlap(cinfo_b)
        Dim x_overlap_b = cinfo_b.local_overlap(cinfo_a)
        Dim overall_tangent = cinfo_b.center - cinfo_a.center
        Dim overall_angle = np.arctan2(overall_tangent(1), overall_tangent(0))
        Dim delta_angle = Math.Max(angle_dist(cinfo_a.angle, overall_angle), angle_dist(cinfo_b.angle, overall_angle)) * 180 / Math.PI
        Dim x_overlap = Math.Max(x_overlap_a, x_overlap_b)
        Dim dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

        If dist > EDGE_MAX_LENGTH OrElse x_overlap > EDGE_MAX_OVERLAP OrElse delta_angle > EDGE_MAX_ANGLE Then
            Return Nothing
        Else
            Dim score = dist + delta_angle * EDGE_ANGLE_COST
            Return Tuple.Create(score, cinfo_a, cinfo_b)
        End If
    End Function

    Function make_tight_mask(ByVal contour As VectorOfPoint, ByVal xmin As Object, ByVal ymin As Object, ByVal width As Object, ByVal height As Object) As Object
        Dim tight_mask = np.zeros(Tuple.Create(height, width), dtype:="uint8") ' OK

        Dim tight_object As New Emgu.CV.Util.VectorOfPoint(CInt(xmin * ymin))
        Dim tight_Contour1 As New Mat
        Emgu.CV.CvInvoke.Subtract(contour, tight_object, tight_Contour1)

        'Dim tight_contour = contour - np.array(Tuple.Create(xmin, ymin)) ' .reshape(Tuple.Create(-1, 1, 2))
        Dim Scal As New MCvScalar(1, 1, 1)
        'CvInvoke.DrawContours(tight_mask, New List(Of Object) From {tight_contour}, 0, Scal, -1)
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
        Dim contours_out = New List(Of Object)()

        For i As Integer = 0 To contours.Size - 1                   ' OK 
            Dim rect = CvInvoke.BoundingRectangle(contours(i))      ' OK

            If rect.Width < TEXT_MIN_WIDTH OrElse
                rect.Height < TEXT_MIN_HEIGHT OrElse
                rect.Width < TEXT_MIN_ASPECT * rect.Height Then Continue For ' OK

            Dim tight_mask = make_tight_mask(contours(i), rect.X, rect.Y, rect.Width, rect.Height)
            'If tight_mask.sum(axis:=0).max() > TEXT_MAX_THICKNESS Then Continue For
            contours_out.Add(New ContourInfo(contours(i), rect, tight_mask))
        Next

        If DEBUG_LEVEL >= 2 Then
            visualize_contours(name, small, contours_out)
        End If

        Return contours_out
    End Function

    Function assemble_spans(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal cinfo_list As Object) As Object
        'cinfo_list = cinfo_list.OrderBy(Function(cinfo) cinfo.rect(1)).ToList()
        'Dim candidate_edges = New List(Of Object)()

        'For Each _tup_1 In cinfo_list.[Select](Function(_p_2, _p_3) Tuple.Create(_p_3, _p_2))
        '    Dim i = _tup_1.Item1
        '    Dim cinfo_i = _tup_1.Item2

        '    For Each j In range(i)
        '        Dim edge = generate_candidate_edge(cinfo_i, cinfo_list(j))

        '        If edge IsNot Nothing Then
        '            candidate_edges.append(edge)
        '        End If
        '    Next
        'Next

        'candidate_edges.sort()

        'For Each _tup_2 In candidate_edges
        '    Dim cinfo_a = _tup_2.Item2
        '    Dim cinfo_b = _tup_2.Item3

        '    If cinfo_a.succ Is Nothing AndAlso cinfo_b.pred Is Nothing Then
        '        cinfo_a.succ = cinfo_b
        '        cinfo_b.pred = cinfo_a
        '    End If
        'Next

        'Dim spans = New List(Of Object)()

        'While cinfo_list
        '    Dim cinfo = cinfo_list(0)

        '    While cinfo.pred
        '        cinfo = cinfo.pred
        '    End While

        '    Dim cur_span = New List(Of Object)()
        '    Dim width = 0.0

        '    While cinfo
        '        cinfo_list.remove(cinfo)
        '        cur_span.Add(cinfo)
        '        width += cinfo.local_xrng(1) - cinfo.local_xrng(0)
        '        cinfo = cinfo.succ
        '    End While

        '    If width > SPAN_MIN_WIDTH Then
        '        spans.Add(cur_span)
        '    End If
        'End While

        'If DEBUG_LEVEL >= 2 Then
        '    visualize_spans(name, small, pagemask, spans)
        'End If

        'Return spans
    End Function

    Function sample_spans(ByVal shape As Object, ByVal spans As Object) As Object
        'Dim span_points = New List(Of Object)()

        'For Each span In spans
        '    Dim contour_points = New List(Of Object)()

        '    For Each cinfo In span
        '        Dim yvals = np.arange(cinfo.mask.shape(0)).reshape(Tuple.Create(-1, 1))
        '        Dim totals = (yvals * cinfo.mask).sum(axis:=0)
        '        Dim means = totals / cinfo.mask.sum(axis:=0)
        '        Dim _tup_1 = cinfo.rect(2)
        '        Dim xmin = _tup_1.Item1
        '        Dim ymin = _tup_1.Item2
        '        Dim [step] = SPAN_PX_PER_STEP
        '        Dim start = (means.Count - 1) Mod [step] / 2
        '        contour_points += range(start, means.Count, [step]).[Select](Function(x) Tuple.Create(x + xmin, means(x) + ymin))
        '    Next

        '    contour_points = np.array(contour_points, dtype:="double").reshape(Tuple.Create(-1, 1, 2))
        '    contour_points = pix2norm(shape, contour_points)
        '    span_points.Add(contour_points)
        'Next

        'Return span_points
    End Function

    Function keypoints_from_samples(ByVal name As Object, ByVal small As Mat, ByVal pagemask As Object, ByVal page_outline As Object, ByVal span_points As Object) As Object
        'Dim evec As Object
        'Dim all_evecs As Double() = {0.0, 0.0}
        'Dim all_weights = 0

        'For Each points In span_points
        '    Dim Vec As New MCvScalar(-1, 2)
        '    Dim _tup_1 : CvInvoke.PCACompute(points.reshape(Vec), _tup_1, Nothing, maxComponents:=1)
        '    evec = _tup_1.Item2
        '    Dim weight = np.linalg.norm(points(-1) - points(0))
        '    all_evecs += evec * weight
        '    all_weights += weight
        'Next

        'evec = all_evecs / all_weights
        'Dim x_dir = evec.flatten()

        'If x_dir(0) < 0 Then
        '    x_dir = -x_dir
        'End If

        'Dim y_dir = np.array(New List(Of Object) From {
        '    -x_dir(1),
        '    x_dir(0)
        '})
        'Dim pagecoords = CvInvoke.ConvexHull(page_outline)
        'pagecoords = pix2norm(pagemask.shape, pagecoords.reshape(tuple.Create(-1, 1, 2)))
        'pagecoords = pagecoords.reshape(tuple.Create(-1, 2))
        'Dim px_coords = np.dot(pagecoords, x_dir)
        'Dim py_coords = np.dot(pagecoords, y_dir)
        'Dim px0 = px_coords.min()
        'Dim px1 = px_coords.max()
        'Dim py0 = py_coords.min()
        'Dim py1 = py_coords.max()
        'Dim p00 = px0 * x_dir + py0 * y_dir
        'Dim p10 = px1 * x_dir + py0 * y_dir
        'Dim p11 = px1 * x_dir + py1 * y_dir
        'Dim p01 = px0 * x_dir + py1 * y_dir
        'Dim corners = np.vstack(Tuple.Create(p00, p10, p11, p01)).reshape(Tuple.Create(-1, 1, 2))
        'Dim ycoords = New List(Of Object)()
        'Dim xcoords = New List(Of Object)()

        'For Each points In span_points
        '    Dim pts = points.reshape(Tuple.Create(-1, 2))
        '    px_coords = np.dot(pts, x_dir)
        '    py_coords = np.dot(pts, y_dir)
        '    ycoords.Add(py_coords.mean() - py0)
        '    xcoords.Add(px_coords - px0)
        'Next

        'If DEBUG_LEVEL >= 2 Then
        '    visualize_span_points(name, small, span_points, corners)
        'End If

        'Return Tuple.Create(corners, np.array(ycoords), xcoords)
    End Function

    Function visualize_contours(ByVal name As String, ByVal small As Mat, ByVal cinfo_list As Object) As Object
        'Dim cinfo As Object
        'Dim j As Object
        'Dim regions = np.zeros_like(small)

        'For Each _tup_1 In cinfo_list.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
        '    j = _tup_1.Item1
        '    cinfo = _tup_1.Item2
        '    CvInvoke.DrawContours(regions, New List(Of Object) From {
        '        cinfo.contour
        '    }, 0, CCOLORS(j Mod CCOLORS.Count), -1)
        'Next

        'Dim mask = regions.max(axis:=2) <> 0
        'Dim display = small.Clone()
        'display(mask) = display(mask) / 2 + regions(mask) / 2

        'For Each _tup_2 In cinfo_list.[Select](Function(_p_3, _p_4) Tuple.Create(_p_4, _p_3))
        '    Dim Col As New MCvScalar(255, 255, 255)
        '    j = _tup_2.Item1
        '    cinfo = _tup_2.Item2
        '    Dim color = CCOLORS(j Mod CCOLORS.Count)
        '    color = tuple.Create(color.[Select](Function(c) c / 4))
        '    CvInvoke.Circle(display, fltp(cinfo.center), 3, Col, 1, LineType.AntiAlias)
        '    CvInvoke.Line(display, fltp(cinfo.point0), fltp(cinfo.point1), Col, 1, LineType.AntiAlias)
        'Next

        'debug_show(name, 1, "contours", display)
    End Function

    Function visualize_spans(ByVal name As Object, ByVal small As Object, ByVal pagemask As Object, ByVal spans As Object) As Object
        Dim regions = np.zeros_like(small)

        For Each _tup_1 In spans.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
            Dim i = _tup_1.Item1
            Dim span = _tup_1.Item2
            Dim contours = span.[Select](Function(cinfo) cinfo.contour)
            CvInvoke.DrawContours(regions, contours, -1, CCOLORS(i * 3 Mod CCOLORS.Count), -1)
        Next

        Dim mask = regions.max(axis:=2) <> 0
        Dim display = small.copy()
        display(mask) = display(mask) / 2 + regions(mask) / 2
        ' display[pagemask == 0] /= 4;
        debug_show(name, 2, "spans", display)
    End Function

    Function visualize_span_points(ByVal name As Object, ByVal small As Object, ByVal span_points As Object, ByVal corners As Object) As Object
        Dim display = small.copy()
        Dim Col As New MCvScalar(255, 255, 255)

        For Each _tup_1 In span_points.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
            Dim i = _tup_1.Item1
            Dim points = _tup_1.Item2
            points = norm2pix(small.shape, points, False)
            Dim Tup As New MCvScalar(-1, 2)
            Dim _tup_2 = Nothing : CvInvoke.PCACompute(points.reshape(Tup), _tup_2, Nothing, maxComponents:=1)
            Dim mean = _tup_2.Item1
            Dim small_evec = _tup_2.Item2
            Dim dps = np.dot(points.reshape(Tuple.Create(-1, 2)), small_evec.reshape(Tuple.Create(2, 1)))
            Dim dpm = np.dot(mean.flatten(), small_evec.flatten())
            Dim point0 = mean + small_evec * (dps.min() - dpm)
            Dim point1 = mean + small_evec * (dps.max() - dpm)

            For Each point In points
                CvInvoke.Circle(display, fltp(point), 3, CCOLORS(i Mod CCOLORS.Count), -1, LineType.AntiAlias)
            Next

            CvInvoke.Line(display, fltp(point0), fltp(point1), Col, 1, LineType.AntiAlias)
        Next

        CvInvoke.Polylines(display, New List(Of Object) From {norm2pix(small.shape, corners, True)}, True, Col)
        debug_show(name, 3, "span points", display)
    End Function

    Function imgsize(ByVal img As Object) As Object
        Dim _tup_1 = img.shape(2)
        Dim height = _tup_1.Item1
        Dim width = _tup_1.Item2
        Return "{}x{}".format(width, height)
    End Function

    Function make_keypoint_index(ByVal span_counts As Object) As Object
        'Dim nspans = span_counts.Count
        'Dim npts = span_counts.Sum()
        'Dim keypoint_index = np.zeros(Tuple.Create(npts + 1, 2), "int")
        'Dim start = 1

        'For Each _tup_1 In span_counts.[Select](Function(_p_1, _p_2) Tuple.Create(_p_2, _p_1))
        '    Dim i = _tup_1.Item1
        '    Dim count = _tup_1.Item2
        '    Dim [end] = start + count
        '    keypoint_index(start(start + [end]), 1) = 8 + i
        '    start = [end]
        'Next

        'keypoint_index(1, 0) = np.arange(npts) + 8 + nspans
        'Return keypoint_index
    End Function

    Function optimize_params(ByVal name As Object, ByVal small As Object, ByVal dstpoints As Object, ByVal span_counts As Object, ByVal _params As Object) As Object
        Dim display As Object
        Dim projpts As Object
        Dim keypoint_index = make_keypoint_index(span_counts)
        Dim objective As Func(Of Object, Object) = Function(pvec)
                                                       Dim ppts = project_keypoints(pvec, keypoint_index)
                                                       Return np.sum(Math.Pow(dstpoints - ppts, 2))
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
        Dim dst_br = corners(2).flatten()
        Dim dims As Object = np.array(rough_dims)
        'Dim objective As Func(Of Object, Object) = Function(dims)
        '                                               Dim proj_br = project_xy(dims, _params)
        '                                               Return np.sum(Math.Pow(dst_br - proj_br.flatten(), 2))
        '                                           End Function

        'Dim res = New scipy.optimize().minimize(objective, dims, method:="Powell") ' Todo - Link to Function needed
        'dims = res.x
        'Return dims
    End Function

    Function remap_image(ByVal name As Object, ByVal img As Object, ByVal small As Object, ByVal page_dims As Object, ByVal _params As Object) As Object
        Dim height = 0.5 * page_dims(1) * OUTPUT_ZOOM * img.shape(0)
        height = round_nearest_multiple(height, REMAP_DECIMATE)
        Dim width = round_nearest_multiple(height * page_dims(0) / page_dims(1), REMAP_DECIMATE)
        Dim height_small = height / REMAP_DECIMATE
        Dim width_small = width / REMAP_DECIMATE
        Dim page_x_range = np.linspace(0, page_dims(0), width_small)
        Dim page_y_range = np.linspace(0, page_dims(1), height_small)
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
        Dim remapped As Mat = Nothing : CvInvoke.Remap(img_gray, remapped, image_x_coords, image_y_coords, Inter.Cubic, BorderType.Replicate, Nothing)
        Dim thresh As Mat = Nothing : CvInvoke.AdaptiveThreshold(remapped, thresh, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, ADAPTIVE_WINSZ, 25)

        'Dim pil_image = Image.fromarray(thresh)
        'pil_image = pil_image.convert("1")
        'Dim threshfile = name & "_thresh.png"
        'pil_image.save(threshfile, dpi:=Tuple.Create(OUTPUT_DPI, OUTPUT_DPI))

        'Return threshfile
    End Function

    Function main() As Object
        Dim outfiles = New List(Of Object)()

        'For Each imgfile In Environment.CommandLine
        Dim imgfile As String = "Test.jpg"  ' OK
        Dim img = CvInvoke.Imread(imgfile)  ' OK
        Dim small = resize_to_screen(img)   ' OK
        Dim name As String = imgfile        ' OK
        Dim basename As String = Environment.CurrentDirectory

        Dim _tup_2 = get_page_extents(small)
        Dim pagemask As Mat = _tup_2.Item1
        Dim page_outline = _tup_2.Item2
        Dim cinfo_list = get_contours(name, small, pagemask, "text")
        Dim spans = assemble_spans(name, small, pagemask, cinfo_list)

        If spans.Count < 3 Then
            Console.WriteLine("  detecting lines because only", spans.Count, "text spans")
            cinfo_list = get_contours(name, small, pagemask, "line")
            Dim spans2 = assemble_spans(name, small, pagemask, cinfo_list)
            If spans2.Count > spans.Count Then spans = spans2
        End If

        If spans.Count < 1 Then Return Nothing
        Dim span_points = sample_spans(small.Size, spans)
        Dim _tup_3 = keypoints_from_samples(name, small, pagemask, page_outline, span_points)
        Dim corners = _tup_3.Item1
        Dim ycoords = _tup_3.Item2
        Dim xcoords = _tup_3.Item3
        Dim _tup_4 = get_default_params(corners, ycoords, xcoords)
        Dim rough_dims = _tup_4.Item1
        Dim span_counts = _tup_4.Item2
        Dim _params = _tup_4.Item3
        'Dim dstpoints = np.vstack(Tuple.Create(corners(0).reshape(Tuple.Create(1, 1, 2))) + tuple.Create(span_points))
        'Dim params = optimize_params(name, small, dstpoints, span_counts, _params)
        'Dim page_dims = get_page_dims(corners, rough_dims, _params)
        'Dim outfile = remap_image(name, img, small, page_dims, _params)
        'outfiles.Add(outfile)
        'Next
    End Function

    Function Min()
        Return Nothing
    End Function
    Function Max()
        Return Nothing
    End Function
End Module

Class np
    Shared Function zeros_like()
        'Throw New NotImplementedException
        Return Nothing
    End Function
    Shared Function arctan2(ByVal a As Double, ByVal b As Double)
        'Throw New NotImplementedException
        Return Math.Atan2(a, b)
    End Function
    Shared Function reshape()
        'Throw New NotImplementedException
        Return Nothing
    End Function
    Shared Function arange()
        'Throw New NotImplementedException
        Return Nothing
    End Function
    Shared Function array(ByVal a As Object, Optional ByVal dtype As Object = Nothing)
        Return Mat.Zeros(a.item1, a.item2, DepthType.Cv8U, 1)
        Return Nothing
    End Function
    Shared Function range()
        'Throw New NotImplementedException : 
        Return Nothing
    End Function
    Shared Function sum()
        'Throw New NotImplementedException : 
        Return Nothing
    End Function
    Shared Function dot()
        'Throw New NotImplementedException : 
        Return Nothing
    End Function
    Shared Function vstack()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function hstack()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function linspace()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function linalg()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function ones(ByVal a As Object, Optional ByVal dtype As Object = Nothing)
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function zeros(ByVal a As Object, Optional ByVal dtype As Object = Nothing)
        Return Mat.Zeros(a.item1, a.item2, DepthType.Cv8U, 1)
    End Function
    Shared Function meshgrid()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function polyval()
        Throw New NotImplementedException : Return Nothing
    End Function
    Shared Function minimum(ByVal a As Mat, ByVal b As Mat) As Mat
        Dim c As New Mat : Emgu.CV.CvInvoke.Min(a, b, c) : Return c
    End Function
End Class
Class scipy
    Class optimize
        Function minimize(ByVal a, ByVal b, ByVal method)
            Throw New NotImplementedException : Return Nothing
        End Function
    End Class
End Class