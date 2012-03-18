/*
 * TestFunction7.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */		

/**
 * TestFunctionAbstract extended as the formula 2.
 *  All code released under BSD licensing.
 * 
 * @author Ruthann Sudman
 * @version 0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class TestFunction7 extends TestFunctionAbstract {
	
	/**
	 * Implementation of the abstract method f with the function 2.
	 * 
	 * @param x	Value used to evaluate the function with.
	 * exceptions No known exceptions.
	 */
	public double f(double x) { 
		
		try {
			return 2;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in TestFunction7 f!" + e.getMessage());
			return 0;
		}
	}
}
