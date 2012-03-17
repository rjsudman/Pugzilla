/*
 * TestFunction3.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/* 
 * The TestFunction class implemented as (x-2)*(x-5)/10.
 */
public class TestFunction3 extends TestFunctionAbstract {
	
	public double f(double x) { 
		return (x-2)*(x-5)/10;
	}
}