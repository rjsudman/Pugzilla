/*
 * TestFunction6.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */		

/**
 * TestFunctionAbstract extended as the formula 5+0.8*x+0.3*x*x+Math.sin(x).
 *  All code released under BSD licensing.
 * 
 * @author Ruthann Sudman
 * @version 0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class TestFunction6 extends TestFunctionAbstract {
	
	/**
	 * Implementation of the abstract method f with the function 5+0.8*x+0.3*x*x+Math.sin(x).
	 * 
	 * @param x	Value used to evaluate the function with.
	 * exceptions No known exceptions.
	 */
	public double f(double x) { 
		
		try {
			return 5+0.8*x+0.3*x*x+Math.sin(x);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in TestFunction6 f!" + e.getMessage());
			return 0;
		}
	}
}
