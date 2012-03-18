/*
 * TestFunction4.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/**
 * TestFunctionAbstract extended as the formula (x-2)*(x-5).
 *  All code released under BSD licensing.
 * 
 * @author Ruthann Sudman
 * @version 0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class TestFunction4 extends TestFunctionAbstract {
	
	/**
	 * Implementation of the abstract method f with the function (x-2)*(x-5).
	 * 
	 * @param x	Value used to evaluate the function with.
	 * exceptions No known exceptions.
	 */
	public double f(double x) { 
		
		try {
			return (x-2)*(x-5);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in TestFunction4 f!" + e.getMessage());
			return 0;
		}
	}
}
