/*
 * TestFunctin.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/* 
 * The TestFunction class implements the concept of a function.
 */
public class TestFunction {

	// Global Variables
	private int myRows;
	private int myCols;
	private float[][] myData;
	
	// TestFunction Constructor
	public TestFunction(int rows, int columns) {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable
		
		this.myRows = rows;
		this.myCols= columns;
		this.myData = new float[this.myRows][this.myCols];
		for (r=0; r<this.myRows; r++) {
			for (c=0; c<this.myCols; c++) {
				this.myData[r][c] = 0;
			}
		}
	}
}