/*
 * RunMe.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */

/* 
 * RunMe is a used to demonstrate the functionality of
 * the mathematical library.
 */
public class RunMe {
	
	public static void main (String[] args) {
        
		/* Test 1:  Inverse Matrix 
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
		A.printMe();	// The original matrix
		B.printMe();	// The inverse matrix
		C.printMe();	// The identity matrix 
		
		/* Test 2:	Functions */
		TestFunction Y = new TestFunction();
		TestFunction2 Z = new TestFunction2();;
		System.out.println("(x-2)*(x+8) solve Newton:" + Y.solve_newton(0.5));
		System.out.println("(x-2)*(x+8) Newton optimized: " + Y.optimize_newton(0.5));
		System.out.println("(x-1)*(x+3) solve Newton:" + Z.solve_newton(0.5));
		System.out.println("(x-1)*(x+3) Newton optimized: " + Z.optimize_newton(0.5));	
	}
}



