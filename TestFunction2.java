/*
 * TestFunction2.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */		

/**
 * TestFunctionAbstract extended as the formula x*x-5.0*x.
 *  All code released under BSD licensing.
 * 
 * @author Ruthann Sudman
 * @version 0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class TestFunction2 extends TestFunctionAbstract {
	
	/**
	 * Implementation of the abstract method f with the function x*x-5.0*x.
	 * 
	 * @param x	Value used to evaluate the function with.
	 * exceptions No known exceptions.
	 */
	public double f(double x) { 
		return x*x-5.0*x;
	}
}
