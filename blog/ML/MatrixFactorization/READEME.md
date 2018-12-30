## [A Gentle Introduction to Matrix Factorization for Machine Learning](https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/)

Matrix Factorization(행렬 분해)는 복잡한 행렬 연산을 쉽게 계산할 수 있도록 구성부분으로 줄이는 방법입니다

1. What is a Matrix Decomposition?

    Matrix decomposition은 복잡한 수의 계산을 단순하게 만드는 방법으로
    가장많이 쓰이는 것은 LU, QR Matrix Decomposition이다.
    
2. LU Matrix Decomposition

    A = LU
      
        A: 분해 대상 정사각 행렬(square matrix)
        L: 아래 삼각형 행렬(lower triangle matrix)
        U: 위 삼각형 행렬(higer triangle matrix)
        
    LU 분해는 반복적인 수치 프로세스를 사용하며, 
    분해되지 않거나 쉽게 분해되는 경우에는 사용이 어렵다
   
    실제적으로 더 안정된방법은 LUP분해(부분피벗을 사용한 LU분해)이다.
   
    A = P.L.U
    


3. QR Matrix Decomposition

    QR Decomposition 은 m X n 행렬에 대한 것으로 행렬을 Q, R 구성 요소로 분해한다.
    
    A = QR
    
        A: 분해 대상 행렬
        Q: 크기가 m X m 인 행렬
        R: 크기가 m X n 인 삼각 행렬
        

4. Cholesky Decomposition

    Cholesky Decomposition은 모든 값이 0보다 큰 정사각행렬에 대한 것이다.
    
    A = L.L^T

    A = U^T.U
    
        L: lower triangle matrix
        U: Upper triangle matrix
        
    Cholesky decomposition은 선형회귀에 대한 최소제곱근 및 시뮬레이션/ 최적화 방법에 사용한다               
    
    
