/*
 * TestFunction6.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/* 
 * The TestFunction class implemented as 5+0.8*x+0.3*x*x+Math.sin(x).
 */
public class TestFunction6 extends TestFunctionAbstract {
	
	public double f(double x) { 
		return 5+0.8*x+0.3*x*x+Math.sin(x);
	}
}