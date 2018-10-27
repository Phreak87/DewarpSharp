Imports Emgu.CV
Imports Emgu.CV.CvEnum
Imports Emgu.CV.Util
Imports Emgu.CV.Structure

Class np


#Region "Flatten"
    Shared Function Flatten(ByVal A As Single())
        Return A
    End Function
    Shared Function Flatten(ByVal A As Integer(,))
        Return Accord.Math.Matrix.Flatten(A)
    End Function
    Shared Function Flatten(ByVal A As Double(,))
        Return Accord.Math.Matrix.Flatten(A)
    End Function
    Shared Function Flatten(ByVal A As Double())
        Return A
    End Function
#End Region


    Shared Function MatToInteger(ByVal Mat As Mat) As Byte(,)
        Dim Ret As Byte(,) = Accord.Math.Matrix.Reshape(Mat.GetData, Mat.Height, Mat.Width, Accord.Math.MatrixOrder.Default)
        Return Ret
    End Function
    Shared Function add(ByVal a As Double(), ByVal b As Double()) As Double()
        Return {a(0) + b(0), a(1) + b(1)}
    End Function
    Shared Function mul(ByVal a As Double(), ByVal b As Double) As Double()
        Return {a(0) * b, a(1) * b}
    End Function
    Shared Function mul(ByVal a As Single (), ByVal b As Double) As Double()
        Return {a(0) * b, a(1) * b}
    End Function
    Shared Function mul(ByVal a As List(Of Integer()), ByVal B As List(Of Integer())) As List(Of Integer())
        Dim Ret As New List(Of Integer())
        Ret.AddRange(B)
        For Row As Integer = 0 To B.Count - 1
            For Col As Integer = 0 To B(0).Length - 1
                Ret(Row)(Col) = Ret(Row)(Col) * a(Row)(0)
            Next
        Next
        Return Ret
    End Function
    Shared Function mulLine(a As Byte(,))
        Dim RES As Byte(,) = a.Clone
        For i As Integer = 0 To UBound(a, 1)
            For i2 As Integer = 0 To UBound(a, 2)
                RES(i, i2) = a(i, i2) * i
            Next
        Next
        Return RES
    End Function

    Shared Function Mean(ByVal a As Double()) As Double
        Dim Sum As Double = 0
        For Each Entry In a
            Sum += Entry
        Next
        Return Sum / a.Count
    End Function

    Shared Function Substract(ByVal a As Double(), ByVal B As Double) As Double()
        Dim Res As New List(Of Double)
        For Each Entry In a
            Res.Add(Entry - B)
        Next
        Return Res.ToArray
    End Function
    Shared Function Substract(ByVal a As Double(), ByVal B As Double()) As Double()
        If a.Count = 2 And B.Count = 2 Then
            Return {a(0) - B(0), a(1) - B(1)}
        Else
            Throw New Exception("Not a 2-Value Double Array")
        End If
    End Function

    Shared Function zeros_like(ByVal a As Mat)
        Return np.zeros(Tuple.Create(a.Height, a.Width))
    End Function
    Shared Function arctan2(ByVal a As Double, ByVal b As Double)
        Return Math.Atan2(a, b)
    End Function

    Shared Function reshape(ByVal a As Byte(), ByVal b As Tuple(Of Integer, Integer))
        Return Accord.Math.Matrix.Reshape(a, b.Item1, b.Item2, Accord.Math.MatrixOrder.Default)
    End Function
    Shared Function reshape(ByVal a As Integer(), ByVal b As Tuple(Of Integer, Integer, Integer))
        If b.Item1 = -1 And b.Item2 = 1 Then
            Dim Ret As New List(Of Integer())
            For Each entry In a
                Ret.Add({entry})
            Next
            Return Ret
        End If
        Return Nothing
    End Function
    Shared Function reshape(ByVal a As Integer(,), ByVal b As Tuple(Of Integer, Integer, Integer))
        If b.Item1 = -1 And b.Item2 = 1 Then
            Dim Ret As New List(Of Integer())
            For Each entry In a
                Ret.Add({entry})
            Next
            Return Ret
        End If
        Return Nothing
    End Function


    Shared Function arange(ByVal a As Object)
        Dim Res As New List(Of Integer)
        For i As Integer = 1 To a - 1
            Res.Add(i)
        Next
        Return Res.ToArray
    End Function

    Shared Function array(ByVal a As List(Of Double()), Optional ByVal dtype As Object = Nothing)
        'Return Mat.Zeros(a.item1, a.item2, DepthType.Cv8U, 1)
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

    Shared Function div(a As Double(), b As Double) As Single()
        Dim Res(UBound(a, 1)) As Single
        For Col As Integer = 0 To UBound(a, 1)
            Res(Col) = a(Col) / b
        Next
        Return Res
    End Function
    Shared Function div(a As Integer(), b As Integer())
        Dim Res(UBound(a, 1)) As Double
        For Col As Integer = 0 To UBound(a, 1)
            Res(Col) = a(Col) / b(Col)
        Next
        Return Res
    End Function
    Shared Function sum(a As Byte(,), b As Integer)
        Dim Res(UBound(a, 2)) As Integer
        For Col As Integer = 0 To UBound(a, 2)
            Dim Line = Accord.Math.Matrix.GetColumn(a, Col)
            For i As Integer = 0 To Line.Length - 1
                Res(Col) += Line(i)
            Next
        Next
        Return Res
    End Function
    Shared Function sum(ByVal a As List(Of Integer), ByVal b As Integer)
        Dim Res As Integer = 0
        If b = 1 Then ' Achse 1
            For i As Integer = 0 To a.Count - 1
                Res += a(i)
            Next
        End If
        Return Res
    End Function
    Shared Function sum(ByVal a As List(Of Integer()), ByVal b As Integer)
        Dim Res As New List(Of Integer)
        If b = 0 Then ' Achse 0
            For i As Integer = 0 To a(0).Length - 1
                Dim ColSum As Integer = 0
                For Each Row In a
                    ColSum += Row(i)
                Next
                Res.Add(ColSum)
            Next
        End If
        Return Res
    End Function

#Region "Dot-Product"
    Shared Function dot(ByVal a As List(Of Double()), ByVal b As Single())
        Dim Ret As New List(Of Double)

        Dim B1 As New Matrix(Of Double)(1, 2)
        B1(0, 0) = b(0)
        B1(0, 1) = b(1)

        For i As Integer = 0 To a.Count - 1
            Dim A1 As New Matrix(Of Double)(1, 2)
            A1(0, 0) = a(i)(0)
            A1(0, 1) = a(i)(1)
            Dim C1 = A1.GetInputArray.GetMat.Dot(B1)
            A1.Dispose() : Ret.Add(C1)
        Next

        B1.Dispose()
        Return Ret.ToArray
    End Function
    Shared Function dot(ByVal a As PointF(), ByVal b As Single())
        Dim Ret As New List(Of Double)
        For i As Integer = 0 To a.Count - 1
            Dim A1 As New VectorOfDouble({a(i).X, a(i).Y})
            Dim B1 As New VectorOfDouble({b(0), b(1)})
            Dim C1 = A1.GetInputArray.GetMat.Dot(B1)
            Ret.Add(C1)
            A1.Dispose()
            B1.Dispose()
        Next
        Return Ret.ToArray
    End Function
    Shared Function dot(ByVal a As Double(), ByVal b As Double())
        Dim A1 As New VectorOfDouble(a)
        Dim B1 As New VectorOfDouble(b)
        Dim C1 = A1.GetInputArray.GetMat.Dot(B1)
        A1.Dispose()
        B1.Dispose()
        Return C1
    End Function
#End Region


    Shared Function vstack(a As Object)
        Return a
    End Function
    Shared Function hstack(a As Object)
        Return a
    End Function
    Shared Function linspace(a As Integer, b As Integer, c As Integer)
        Dim Steps As Integer = (b - a) / c
        Dim Ret As New List(Of Double)
        For i As Integer = a To b Step Steps
            Ret.Add(a + (i * Steps))
        Next
        Return Ret
    End Function
    Public Class linalg
        Shared Function norm(ByVal a As Double()) As Double
            Return MathNet.Numerics.LinearAlgebra.Vector(Of Double).Build.SparseOfArray(a).L2Norm
        End Function
    End Class

    Shared Function ones(ByVal a As Object, Optional ByVal dtype As Object = Nothing)
        Return Mat.Ones(a.item1, a.item2, DepthType.Cv8U, 1)
    End Function
    Shared Function zeros(ByVal a As Object, Optional ByVal dtype As DepthType = DepthType.Cv8U, Optional Channels As Integer = 1)
        Return Mat.Zeros(a.item1, a.item2, dtype, Channels)
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


    Shared Function ToMatrix(Tup As Tuple(Of Double(), Double(), Double(), Double())) As Matrix(Of Double)
        Dim RES As New Matrix(Of Double)(4, Tup.Item1.Length)
        RES.Data(0, 0) = Tup.Item1(0)
        RES.Data(0, 1) = Tup.Item1(1)
        RES.Data(1, 0) = Tup.Item2(0)
        RES.Data(1, 1) = Tup.Item2(1)
        RES.Data(2, 0) = Tup.Item3(0)
        RES.Data(2, 1) = Tup.Item3(1)
        RES.Data(3, 0) = Tup.Item4(0)
        RES.Data(3, 1) = Tup.Item4(1)
        Return RES
    End Function
    Shared Function ToMatrix(Input As Double(,)) As Matrix(Of Double)
        Dim A2L As Integer = UBound(Input, 2) + 1
        Dim RES As New Matrix(Of Double)(Input.Length / A2L, A2L)
        For i1 As Integer = 0 To (Input.Length / A2L) - 1
            For i2 As Integer = 0 To A2L - 1
                RES.Data(i1, i2) = Input(i1, i2)
            Next
        Next
        Return Res
    End Function
    Shared Function ToMatrix(Input As Double()()) As Matrix(Of Double)
        Dim A2L As Integer = UBound(Input, 2) + 1
        Dim RES As New Matrix(Of Double)(Input.Length / A2L, A2L)
        For i1 As Integer = 0 To (Input.Length / A2L) - 1
            For i2 As Integer = 0 To A2L - 1
                RES.Data(i1, i2) = Input(i1)(i2)
            Next
        Next
        Return RES
    End Function

End Class