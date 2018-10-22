Imports Emgu.CV.Util
Imports Emgu.CV

Public Class ContourInfo
    Property contour As VectorOfPoint
    Property rect
    Property mask As Mat
    Property center
    Property tangent
    Property angle
    Property local_xrng As Tuple(Of Double, Double)
    Property point0 As Double()
    Property point1 As Double()
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
        Dim clx As New List(Of Double)
        For Each entry In Me.contour.ToArray
            clx.Add(proj_x(entry))
        Next
        Dim lxmin = clx.Min()
        Dim lxmax = clx.Max()
        Me.local_xrng = Tuple.Create(lxmin, lxmax)

        Me.point0 = np.add(np.mul(tangent, lxmin), center)
        Me.point1 = np.add(np.mul(tangent, lxmax), center)
        Me.pred = Nothing
        Me.succ = Nothing
    End Sub

    Public Overridable Function proj_x(ByVal point As Drawing.Point) As Object
        Return proj_x({CDbl(point.X), CDbl(point.Y)})
    End Function
    Public Overridable Function proj_x(ByVal point As Double()) As Object
        Return np.dot(Me.tangent, np.Substract({CDbl(point(0)), CDbl(point(1))}, center))
    End Function
    <DebuggerStepThrough()>
    Public Overridable Function local_overlap(ByVal other As ContourInfo) As Object
        Dim xmin As Double = Me.proj_x(other.point0) ' OK
        Dim xmax As Double = Me.proj_x(other.point1) ' OK
        Return interval_measure_overlap(Me.local_xrng, Tuple.Create(xmin, xmax)) ' OK
    End Function
End Class
