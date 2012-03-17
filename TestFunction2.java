/*
 * TestFunction2.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/* 
 * The TestFunction class implemented as x*x-5.0*x.
 */
public class TestFunction2 extends TestFunctionAbstract {
	
	public double f(double x) { 
		return x*x-5.0*x;
	}
}
