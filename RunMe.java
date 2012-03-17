/*
 * RunMe.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */

import java.text.DecimalFormat;


/* 
 * RunMe is a used to demonstrate the functionality of
 * the mathematical library.
 */
public class RunMe {
	
	// Variable declaration
	private static TestFunction Y = new TestFunction();		// Test function (x-2)*(x+8)
	private static TestFunction2 Z = new TestFunction2();	// Test function (x-1)*(x+3)
	private static TestFunction3 P = new TestFunction3();	// Test function (x-2)*(x-5)/10
	private static TestFunction4 Q = new TestFunction4();	// Test function (x-2)*(x-5)
	private static LinearAlgebra LA = new LinearAlgebra();	// A LinearAlgebra object
	private static DecimalFormat twoD = new DecimalFormat("#.0");
	private static DecimalFormat twelveD = new DecimalFormat("0.000000000000");
	
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
	
	// BROKEN
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
		TestMatrix W = new TestMatrix(3,1);
		int k;
		
		for(k=90; k<=00; k++) {
			W = Y.fit_least_squares((double) k,5+0.8*k+0.3*k*k+Math.sin(k),2.0);
		}
		System.out.println("BROKEN****fit_least_squares****");
		System.out.println("(x-2)*(x+8) fit_least_squares x=1.0 y=3.0 z=0.5: ");
		W = Y.fit_least_squares(1.0,3.0,0.5);
		W.printMe();
		System.out.println("     Expected Result: BROKEN");
		System.out.println("(x-1)*(x+3) fit_least_squares x=1.0 y=3.0 z=0.5: ");
		W =Z.fit_least_squares(1.0,3.0,0.5);
		W.printMe();
		System.out.println("     Expected Result: BROKEN");
		System.out.println("");
	}
	
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
	
	public static void main (String[] args) {
		
		Test1();	// TestMatrix, exp
		Test2();	// Cholesky, is_almost_symmetric (part of Cholesky), is_almost_zero
		Test3();	// Markovitz
		Test4();	// fit_least_squares BROKEN
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



