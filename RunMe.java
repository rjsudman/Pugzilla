/*
 * RunMe.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */

import java.text.DecimalFormat;


/**
 * Used to demonstrate the functionality of the mathematical library. 
 * Algorithms originally created in Python by Massimo Di Pierro and ported to 
 * Java.  All code released under BSD licensing.
 * 
 * @author					Ruthann Sudman
 * @version					0.1
 * @see TestMatrix
 * @see LinearAlgebra
 * @see TestFunctionAbstract
 * @see TestFunction
 * @see TestFunction2
 * @see TestFunction3
 * @see TestFunction4
 * @see TestFunction5
 * @see TestFunction6
 * @see TestFunction7
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class RunMe {
		
	// Variable declaration
	private static TestFunction Y = new TestFunction();		// Test function x*x-5.0*x
	private static TestFunction2 Z = new TestFunction2();	// Test function x*x-5.0*x
	private static TestFunction3 P = new TestFunction3();	// Test function (x-2)*(x-5)/10
	private static TestFunction4 Q = new TestFunction4();	// Test function (x-2)*(x-5)
	/*private static TestFunction5 F1 = new TestFunction5();	// Function x
	private static TestFunction6 F2 = new TestFunction6(); // Function 5+0.8*x+0.3*x*x+Math.sin(x)
	private static TestFunction7 F3 = new TestFunction7();	// Function 2 */
	private static LinearAlgebra LA = new LinearAlgebra();	// A LinearAlgebra object
	private static DecimalFormat twoD = new DecimalFormat("#.0");
	private static DecimalFormat twelveD = new DecimalFormat("0.000000000000");
	
	/**
	 * Tests inverse matrix as implemented in class using c++.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestMatrix
	 * @see TestMatrix#invMatrix()
	 * @see TestMatrix#mulMatrix(TestMatrix)
	 */
	public static void Test1() {
		/* Test 1:  Inverse Matrix 
		 * This test mirrors the output presented in class
		 * The expected output is:
		 * [ {1.0, 2.0, 3.0} {1.0, 0.0, 3.0} {2.0, 2.0, 4.0} ]
		 * [ {-1.5, -0.5, 1.5} {0.5, -0.5, 0.0} {0.5, 0.5, -0.5} ]
		 * [ {1.0, 0.0, 0.0} {0.0, 1.0, 0.0} {0.0, 0.0, 1.0} ]
		 */
        
		// Variable declaration
		TestMatrix A = new TestMatrix(3,3);	// The original matrix
		TestMatrix B;						// The inverse matrix
		TestMatrix C;						// The identity matrix
		
		// Variable initialization
		A.changeMe(0,0,1); 
		A.changeMe(0,1,2); 
		A.changeMe(0,2,3);
		A.changeMe(1,0,1); 
		A.changeMe(1,1,0); 
		A.changeMe(1,2,3);
		A.changeMe(2,0,2); 
		A.changeMe(2,1,2); 
		A.changeMe(2,2,4); 
		
		// Testing
		B = A.invMatrix();
		C = A.mulMatrix(B);
		
		// Printing Results
		System.out.println("****TestMatrix****");
		System.out.print("The original matrix A: ");
		A.printMe();	// The original matrix
		System.out.println("      Expected Result: [ [1.0, 2.0, 3.0] [1.0, 0.0, 3.0] [2.0, 2.0, 4.0] ]");
		System.out.print("The inverse matrix B: ");
		B.printMe();	// The inverse matrix
		System.out.println("     Expected Result: [ [-1.5, -0.5, 1.5] [0.5, -0.5, 0.0] [0.5, 0.5, -0.5] ]");
		System.out.print("The identity matrix A*B: ");
		C.printMe();	// The identity matrix 
		System.out.println("         Expected Result:[ [1.0, 0.0, 0.0] [0.0, 1.0, 0.0] [0.0, 0.0, 1.0] ]");
		System.out.println("");
	}
	
	/**
	 * Tests Cholesky as implemented in test096 from Massimo Ei Pierro's
	 * numeric.py. 
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestMatrix
	 * @see LinearAlgebra
	 * @see LinearAlgebra#Cholesky(TestMatrix)
	 */
	public static void Test2() {
		/* Test 2: Cholesky, is_almost_symmetric, is_almost zero
		 * This test mirrors test096 from numeric.py 
		 * The expected output is:
		 * [[2.0, 0, 0], [1.0, 2.8284271247461903, 0], [0.5, 0.8838834764831843, 3.86894688513554]]
		 * True
		 */
        
		// Variable declaration
		TestMatrix A = new TestMatrix(3,3);	// The original matrix
		TestMatrix C = new TestMatrix(3,2);	// A copy of the original matrix
		
		// Variable initialization
		A.changeMe(0,0,4); 
		A.changeMe(0,1,2); 
		A.changeMe(0,2,1);
		A.changeMe(1,0,2); 
		A.changeMe(1,1,9); 
		A.changeMe(1,2,3);
		A.changeMe(2,0,1); 
		A.changeMe(2,1,3); 
		A.changeMe(2,2,16);
		
		System.out.println("****Cholesky****");
		System.out.print("The original matrix A: ");
		A.printMe();	// The original matrix
		System.out.println("      Expected Result: [ [4.0, 2.0, 1.0] [2.0, 9.0, 3.0] [1.0, 3.0, 16.0] ]");
		System.out.print("Cholesky for matrix A: ");
		C = LA.Cholesky(A);
		C.printMe();
		System.out.println("      Expected Result: [ [2.0, 0.0, 0.0] [1.0, 2.8284271247461903, 0.0] [0.5, 0.8838834764831843, 3.86894688513554] ]");
		System.out.println("if TestMatrix A is_almost_zero: " + LA.is_almost_zero(A));
		System.out.println("               Expected Result: true");
		System.out.println("");
	}
	
	/**
	 * Tests Markovitz as implemented in the orignal Markovitz by Massimo
	 * Di Pierro in numeric.py
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestMatrix
	 * @see LinearAlgebra
	 * @see LinearAlgebra#Markovitz(TestMatrix, TestMatrix, double)
	 * @see LinearAlgebra#getMarkovitzPortfolio()
	 * @see LinearAlgebra#getMarkovitzPortfolioReturn()
	 * @see LinearAlgebra#getMarkovitzPortfolioReturn()
	 */
	public static void Test3() {
		/* Test 3: Markovitz 
		 * This test mirrors the comments for the original Python function
		 * The expected output is:
		 * x [0.5566343042071198], [0.27508090614886727], [0.16828478964401297]]
		 * ret 0.113915857605
		 * risk 0.186747095412
		 */
		// Variable declaration
		double r_free;							// Free rate of return
		TestMatrix cov = new TestMatrix(3,3);	// Covariance TestMatrix
		TestMatrix mu = new TestMatrix(3,1);	// mu TestMatrix
		TestMatrix portfolio;					// Portfolio
		double portfolio_return;				// Return
		double portfolio_risk;					// Risk
		LinearAlgebra Me = new LinearAlgebra();	// Linear Algebra object
        
		// Variable initialization
		cov.changeMe(0,0,0.04); 
		cov.changeMe(0,1,0.006); 
		cov.changeMe(0,2,0.02);
		cov.changeMe(1,0,0.006); 
		cov.changeMe(1,1,0.09); 
		cov.changeMe(1,2,0.06);
		cov.changeMe(2,0,0.02); 
		cov.changeMe(2,1,0.06); 
		cov.changeMe(2,2,0.16);
		mu.changeMe(0, 0,0.10);
		mu.changeMe(1,0,0.12);
		mu.changeMe(2,0,0.15);
		r_free = 0.05;
		
		// Calculate
		Me = Me.Markovitz(mu,cov, r_free);
		portfolio = Me.getMarkovitzPortfolio();
		portfolio_return = Me.getMarkovitzPortfolioReturn();
		portfolio_risk = Me.getMarkovitzPortfolioRisk();
		
		// Printing the results
		System.out.println("****Markovitz****");
		System.out.print("The original matrix A: ");
		cov.printMe();	// The original matrix
		System.out.println("      Expected Result: [ [0.04, 0.006, 0.02] [0.006, 0.09, 0.06] [0.02, 0.06, 0.16] ]");
		System.out.print("Markovitz portfolio for matrix A with r_free=0.05: ");
		portfolio.printMe();
		System.out.println("                                  Expected Result: [ [0.5566343042071198] [0.27508090614886727] [0.16828478964401297] ]");
		System.out.println("Markovitz return for matrix A: " + twelveD.format(portfolio_return));
		System.out.println("              Expected Result: 0.113915857605");
		System.out.println("Markovitz risk for matrix A: " + twelveD.format(portfolio_risk));
		System.out.println("            Expected Result: 0.186747095412");
		System.out.println("");
	}
	
	/**
	 * Tests the condition number and square root for doubles. 
	 * 
	 * @exception ArithmeticException Fails when method is incorrect. The condition number for test matrix is not implemented.
	 * @see	TestMatrix
	 * @see TestMatrix#condition_number()
	 */
	public static void Test35() {
		/* Test 3.5: condition_number
		 * This test mirrors test094 from numeric.py
		 * >>> def f(x): return x*x-5.0*x
		 * >>> print condition_number(f,1)
		 * 0.74999...
		 * >>> A = Matrix.from_list([[1,2],[3,4]])
		 * >>> print condition_number(A)
		 * 21.0
		 */
		
		// Variable declaration
		TestMatrix myMatrix = new TestMatrix(2,2);
		double myCondition;
		
		// Variable initialization
		myMatrix.changeMe(0, 0, 1.0);
		myMatrix.changeMe(0, 1, 2.0);
		myMatrix.changeMe(1, 0, 3.0);
		myMatrix.changeMe(1, 1, 4.0);
		
		System.out.println("***condition_number****");
		System.out.println("Condition number for x*x-5.0*x with x=1: " + twelveD.format(Y.condition_number(1)));
		System.out.println("                        Expected Result: 0.749999999883");
		//System.out.println("Condition number for TestMatrix [[1,2],[3,4]]: " + 
		myCondition = myMatrix.condition_number();
		myCondition = myCondition + 0.0;
		//System.out.println("                              Expected Result: 21.0\n");
		System.out.println("Square root of 4: " + twelveD.format(LA.MySqrt(4)));
		System.out.println(" Expected Result: 2.000000000000\n");
	}
	
	/**
	 * Tests fit least squares for TestFunctionAbstract array of functions.
	 * 
	 * @exception ArithmeticException Not yet implemented.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#fit_least_squares()
	 */
	public static void Test4() {
		/* Test 4: fit_least_squares 
		 * This test mirrors test097 from numeric.py
		 * >>> points = [(k,5+0.8*k+0.3*k*k+math.sin(k),2) for k in range(100)]
		 * >>> a,chi2,fitting_f = fit_least_squares(points,QUADRATIC)
		 * >>> for p in points[-10:]:
		 * ...     print p[0], round(p[1],2), round(fitting_f(p[0]),2)
		 * The expected output is:
		 * 90 2507.89 2506.98
		 * 91 2562.21 2562.08
		 * 92 2617.02 2617.78
		 * 93 2673.15 2674.08
		 * 94 2730.75 2730.98
		 * 95 2789.18 2788.48
		 * 96 2847.58 2846.58
		 * 97 2905.68 2905.28
		 * 98 2964.03 2964.58
		 * 99 3023.5 3024.48
		 */
		// Variable Declaration
		/*TestFunctionAbstract  [] myFunctions = new TestFunctionAbstract[3]; // An array of functions
		int range;
		int k;									// Loop counting variable

		// Initialize Variables
		myFunctions[0] = F1;
		myFunctions[1] = F2;
		myFunctions[3] = F3;
		range = 100;
		*/
		System.out.println("****fit_least_squares****");
		/*System.out.println("(k,5+0.8*k+0.3*k*k+math.sin(k),2) fit_least_squares x=1.0 y=3.0 z=0.5: ");
		W = Y.fit_least_squares(myFunction, range);
		W.printMe();*/
		System.out.println("**fit_least_squares** not implemented.\n");
	}
	
	/**
	 * Tests solve fixed point for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#solve_fixed_point(double)
	 * @see TestFunction3
	 */
	public static void Test5() {
		/* Test 5: solve_fixed_point 
		 * This test mirrors test102 from numeric.py
		 * The expected output is:
		 * 2.0 
		 */
		
		System.out.println("****solve_fixed_point****");
		System.out.println("(x-2)*(x-5)/10 solve_fixed_point x=0.5: " + twoD.format(P.solve_fixed_point(1.0)));
		System.out.println("                       Expected Result: 2.0");
		System.out.println("");
	}
	
	/**
	 * Tests solve bisection for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#solve_bisection(double, double)
	 * @see TestFunction4
	 */
	public static void Test6() {
		/* Test 6: solve_bisection 
		 * This test mirrors test103 from numeric.py
		 * The expected output is:
		 * 2.0
		 */
        
		System.out.println("****solve_bisection****");
		System.out.println("(x-2)*(x-5) solve_bisection a=1.0, b=3.0: " + twoD.format(Q.solve_bisection(1.0,3.0)));
		System.out.println("                         Expected Result: 2.0");
		System.out.println("");
	}

	/**
	 * Tests solve solve newton for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#solve_newton(double)
	 * @see TestFunction4
	 */
	public static void Test7() {
		/* Test 7:	solve_newton 
		 * This test mirrors test104 from numeric.py
		 * The expected output is:
		 * 2.0
		 */
        
		System.out.println("****solve_newton****");
		System.out.println("(x-2)*(x-5) solve_newton x=1.0: " + twoD.format(Q.solve_newton(1.0)));
		System.out.println("               Expected Result: 2.0");
		System.out.println("");
	}
	
	/**
	 * Tests solve secant for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#solve_secant(double)
	 * @see TestFunction4
	 */
	public static void Test8() {
		/* Test 8: solve_secant 
		 * This test mirrors test105 from numeric.py
		 * The expected output is:
		 * 2.0
		 */
        
		System.out.println("****solve_secant****");
		System.out.println("(x-2)*(x-5) solve_secant x=0.5: " + twoD.format(Q.solve_secant(1.0)));
		System.out.println("               Expected Result: 2.0");
		System.out.println("");
	}
	
	/**
	 * Tests solve newton stabilized for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#solve_newton_stabilized(double, double)
	 * @see TestFunction4
	 */
	public static void Test9() {
		/* Test 9: solve_newton_stabilized 
		 * This test mirrors test106 from numeric.py
		 * The expected output is:
		 * 2.0
		 */
        
		System.out.println("****solve_newton_stabilized****");
		System.out.println("(x-2)*(x-5) solve_newton_stabilized a=1.0 b=3.0: " + twoD.format(Q.solve_newton_stabilized(1.0,3.0)));
		System.out.println("                                Expected Result: 2.0");
		System.out.println("");
	}
    
	/**
	 * Tests optimize bisection for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#optimize_bisection(double, double)
	 * @see TestFunction4
	 */
	public static void Test10() {
		/* Test 10: optimize_bisection 
		 * This test mirrors test107 from numeric.py
		 * The expected output is:
		 * 3.5
		 */
        
		System.out.println("****optimize_bisection****");
		System.out.println("(x-2)*(x-5) optimize_bisection a=0.5 b=-1.0: " + twoD.format(Q.optimize_bisection(2.0,5.0)));
		System.out.println("                            Expected Result: 3.5");
		System.out.println("");
	}
	
	/**
	 * Tests optimize newton for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#optimize_newton(double)
	 * @see TestFunction4
	 */
	public static void Test11() {
		/* Test 11: optimize_newton 
		 * This test mirrors test 108 from numeric.py 
		 * The expected output is:
		 * 3.5
		 */
		
		System.out.println("****optimize_newton****");
		System.out.println("(x-2)*(x-5) optimize_newton x=3.0: " + twoD.format(Q.optimize_newton(3.0)));
		System.out.println("                  Expected Result: 3.5");
		System.out.println("");
	}
	
	/**
	 * Tests optimize secant for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#optimize_secant(double)
	 * @see TestFunction4
	 */
	public static void Test12() {
		/* Test 12: optimize_secant 
		 This test mirrors test109 from numeric.py
		 * The expected output is:
		 * 3.5
		 */
        
		System.out.println("****optimize_secant****");
		System.out.println("(x-2)*(x-5) optimize_secant x=3.0: " + twoD.format(Q.optimize_secant(3.0)));
		System.out.println("                  Expected Result: 3.5");
		System.out.println("");
	}
	
	/**
	 * Tests optimize newton stabilized for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#optimize_newton_stabilized(double, double)
	 * @see TestFunction4
	 */
	public static void Test13() {
		/* Test 13: optimize_newton_stabilized 
		 * This test mirrors test106 from numeric.py
		 * The expected output is:
		 * 3.5
		 */
        
		System.out.println("****optimize_newton_stabilized****");
		System.out.println("(x-2)*(x-5) optimize_newton_stabilized a=2.0 b=5.0: " + twoD.format(Q.optimize_newton_stabilized(2.0,5.0)));
		System.out.println("                                   Expected Result: 3.5");
		System.out.println("");
	}
	
	/**
	 * Tests optimize golden search for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#optimize_golden_search(double, double)
	 * @see TestFunction4
	 */
	public static void Test14() {
		/* Test 14: optimize_golden_search 
		 * This test mirrors test106 from numeric.py
		 * The expected output is:
		 * 3.5
		 */
        
		System.out.println("****optimize_golden_search****");
		System.out.println("(x-2)*(x-5) optimize_golden_search a=2.0 b=5.0: " + twoD.format(Q.optimize_golden_search(2.0,5.0)));
		System.out.println("                               Expected Result: 3.5");
		System.out.println("");
	}
	
	/**
	 * Tests first and second derivatives for a function extended from TestFunctionAbstract.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestFunctionAbstract
	 * @see TestFunctionAbstract#f(double)
	 * @see TestFunctionAbstract#Df(double)
	 * @see TestFunctionAbstract#DDf(double)
	 * @see TestFunction2
	 */
	public static void Test15() {
		/* Test 15: Testing derivatives (f(x), Df(x), DDf(x)
		 * This test mirrors test081 in numeric.py
		 * >> def f(x): return x*x-5.0*x
		 * >>> print f(0)
		 * 0.0
		 * >>> f1 = D(f) # first derivative
		 * >>> print f1(0)
		 * -5.0
		 * >>> f2 = DD(f) # second derivative
		 * >>> print f2(0)
		 * 2.00000...
		 */
		
		System.out.println("****Dervatives Test****");
		System.out.println("f(0) x*x-5.0*x: " + Z.f(0.0));
		System.out.println("Expected Result: 0.0");
		System.out.println("Df(0) x*x-5.0*x: " + Z.Df(0.0));
		System.out.println("Expected Result: -5.0");
		System.out.println("DDf(0) x*x-5.0*x: " + twelveD.format(Z.DDf(0.0)));
		System.out.println(" Expected Result: 2.00000000048\n");
	}
	
	/** 
	 * Tests for basic TestMatrix math functionality.
	 * 
	 * @exception ArithmeticException Fails when method is incorrect.
	 * @see TestMatrix
	 * @see TestMatrix#addMatrix(double)
	 * @see TestMatrix#addMatrix(TestMatrix)
	 * @see TestMatrix#changeMe(int, int, double)
	 * @see TestMatrix#condition_number()
	 * @see TestMatrix#copyMe()
	 * @see TestMatrix#divMatrix(double)
	 * @see TestMatrix#invMatrix()
	 * @see TestMatrix#mulMatrix(double)
	 * @see TestMatrix#mulMatrix(TestMatrix)
	 * @see TestMatrix#mulMatrixScalar(TestMatrix)
	 * @see TestMatrix#printMe()
	 * @see TestMatrix#subMatrix(double)
	 * @see TestMatrix#subMatrix(TestMatrix)
	 */
	public static void Test16() {
		/* Test 16: Matrix Math
		 * This test mirrors test087, test089
		 * NOTE unlike the python tests, these tests are NOT additive as below 
		 *     >>> A = Matrix.from_list([[1.0,2.0],[3.0,4.0]])
    	 *>>> print A + A      # calls A.__add__(A)
    	 * [[2.0, 4.0], [6.0, 8.0]]
    	 * >>> print A + 2      # calls A.__add__(2)
    	 * [[3.0, 2.0], [3.0, 6.0]]
    	 * >>> print A - 1      # calls A.__add__(1)
    	 * [[0.0, 2.0], [3.0, 3.0]]
    	 * >>> print -A         # calls A.__neg__()
    	 * [[-1.0, -2.0], [-3.0, -4.0]]
    	 * >>> print 5 - A      # calls A.__rsub__(5)
  	 	 * [[4.0, -2.0], [-3.0, 1.0]]
    	 * >>> b = Matrix.from_list([[1.0],[2.0],[3.0]])
 	  	 * >>> print b + 2      # calls b.__add__(2)
    	 * [[3.0], [4.0], [5.0]]
         * >>> A = Matrix.from_list([[1.0,2.0],[3.0,4.0]])
    	 * >>> print 2*A       # scalar * matrix
    	 * [[2.0, 4.0], [6.0, 8.0]]
    	 * >>> print A*A       # matrix * matrix
    	 * [[7.0, 10.0], [15.0, 22.0]]
    	 * >>> b = Matrix.from_list([[1],[2],[3]])
    	 * >>> print b*b       # scalar product
    	 * 14
         * >>> A = Matrix.from_list([[1,2],[4,9]])
    	 * >>> print 1/A
    	 * [[9.0, -2.0], [-4.0, 1.0]]
    	 * >>> print A/A
    	 * [[1.0, 0.0], [0.0, 1.0]]
    	 * >>> print A/2
    	 * [[0.5, 1.0], [2.0, 4.5]]
		 */
		
		// Variable declaration
		TestMatrix myTest = new TestMatrix(2,2);	// TestMatrix A
		TestMatrix myTest2 = new TestMatrix(1,3);	// TestMatrix B
		TestMatrix myTest3 = new TestMatrix(3,3);	// TestMatrix C
		TestMatrix myTest4 = new TestMatrix(1,3);	// TestMatrix D
		TestMatrix myTest5 = new TestMatrix(2,2);	// TestMatrix E
		TestMatrix x;								// TestMatrix placeholder
		double y;									// Double result placeholder
		
		// Variable initialization
		myTest.changeMe(0, 0, 1.0);
		myTest.changeMe(0, 1, 2.0);
		myTest.changeMe(1, 0, 3.0);
		myTest.changeMe(1, 1, 4.0);
		myTest2.changeMe(0, 0, 1.0);
		myTest2.changeMe(0, 1, 2.0);
		myTest2.changeMe(0, 2, 3.0);
		myTest3.changeMe(0, 0, 1.0);
		myTest3.changeMe(0, 1, 2.0);
		myTest3.changeMe(0, 2, 2.0);
		myTest3.changeMe(1, 0, 4.0);
		myTest3.changeMe(1, 1, 4.0);
		myTest3.changeMe(1, 2, 2.0);
		myTest3.changeMe(2, 0, 4.0);
		myTest3.changeMe(2, 1, 6.0);
		myTest3.changeMe(2, 2, 4.0);
		myTest4.changeMe(0, 0, 3.0);
		myTest4.changeMe(0, 1, 6.0);
		myTest4.changeMe(0, 2, 10.0);
		myTest5.changeMe(0, 0, 1.0);
		myTest5.changeMe(0, 1, 2.0);
		myTest5.changeMe(1, 0, 4.0);
		myTest5.changeMe(1, 1, 9.0);

		System.out.println("****Matrix Math****");
		System.out.print("Matrix A: ");
		myTest.printMe();
		System.out.println("Expected Result: [ [1.0,2.0] [3.0,4.0] ]");
		System.out.print("Matrix B: ");
		myTest2.printMe();
		System.out.println("Expected Result: [ [1.0] [2.0] [3.0] ]");
		System.out.print("Matrix C: ");
		myTest3.printMe();
		System.out.println("Expected Result: [ [1,2,2] [4,4,2] [4,6,4] ]");
		System.out.print("Matrix D: ");
		myTest4.printMe();
		System.out.println("Expected Result: [ [3] [6] [10] ]");
		System.out.print("Matrix E: ");
		myTest5.printMe();
		System.out.println("Expected Result: [ [1,2] [4,9] ]");
		System.out.print("Matrix F: ");
		
		System.out.print("A + A: ");
		x= myTest.addMatrix(myTest);
		x.printMe();
		System.out.println("Expected Result: [[2.0, 4.0], [6.0, 8.0]]");
		System.out.print("A + 2: ");
		x = myTest.addMatrix(2);
		x.printMe();
		System.out.println("Expected Result: [[3.0, 4.0], [5.0, 6.0]]");
		System.out.print("A - 1: ");
		x=myTest.subMatrix(1);
		x.printMe();
		System.out.println("Expected Result: [[0.0, 1.0], [2.0, 3.0]]");
		System.out.print("-A: ");
		x=myTest.mulMatrix(-1.0);
		x.printMe();
		System.out.println("Expected Result: [[-1.0, -2.0], [-3.0, -4.0]]");
		System.out.print("5 - A: ");
		x=myTest.mulMatrix(-1);
		x=x.addMatrix(5);
		x.printMe();
		System.out.println("Expected Result: [[4.0, 3.0], [2.0, 1.0]]");
		System.out.print("B + 2: ");
		x=myTest2.addMatrix(2);
		x.printMe();
		System.out.println("Expected Result: [[3.0], [4.0], [5.0]]");
		System.out.print("2 * E: ");
		x=myTest.mulMatrix(2);
		x.printMe();
		System.out.println("Expected Result: [[2.0, 4.0], [6.0, 8.0]]");
		System.out.print("E * E: ");
		x=myTest.mulMatrix(myTest);
		x.printMe();
		System.out.println("Expected Result: [[7.0, 10.0], [15.0, 22.0]]");
		y=myTest2.mulMatrixScalar(myTest2);
		System.out.println("B * B: " + y);
		System.out.println("Expected Result: 14");
		System.out.print("1 / E: ");
		x=myTest5.invMatrix();
		x.printMe();
		System.out.println("Expected Result: [[9.0, -2.0], [-4.0, 1.0]]");
		System.out.print("E / E: ");
		x=myTest5.invMatrix();
		x=myTest5.mulMatrix(x);
		x.printMe();
		System.out.println("Expected Result: [[1.0, 0.0], [0.0, 1.0]]");
		System.out.print("E / 2: ");
		x=myTest5.divMatrix(2);
		x.printMe();
		System.out.println("Expected Result: [[0.5, 1.0], [2.0, 4.5]]\n");
	}
	
	/**
	 * Runs all test methods.
	 * 
	 * @param args	Default for Java.
	 * @exception ArithmeticException Fails for incorrect methods.
	 */
	public static void main (String[] args) {
		
		Test1();	// TestMatrix, exp
		Test16();	// Testing matrix math
		Test15();	// Testing derivatives (f(x), Df(x), DDf(x)
		Test2();	// Cholesky, is_almost_symmetric (part of Cholesky), is_almost_zero
		Test3();	// Markovitz
		Test35();	// condition_number, MySqrt
		Test4();	// fit_least_squares (not implemented)
		Test5();	// solve_fixed_point
		Test6();	// solve_bisection
		Test7();	// solve_newton
		Test8();	// solve_secant
		Test9();	// solve_newton_stabilized
		Test10();	// optimize_bisection
		Test11();	// opimize_newton
		Test12();	// optimize_secant
		Test13();	// optimize_newton_stabilized
		Test14();	// optimize_golden_search
	}
}



