/*
 * TestMatrix.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

import java.lang.Math;

/* 
 * The TestMatrix class implements basic matrix math, including;
 * addition, subtraction, multiplication and inverse.
 */
public class TestMatrix {
    
	// Global Variables
	private int myRows;
	private int myCols;
	private double [][] myData;
	
	// TestMatrix Constructor
	public TestMatrix(int rows, int columns) {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable
		
		this.myRows = rows;
		this.myCols= columns;
		this.myData = new double[this.myRows][this.myCols];
		for (r=0; r<this.myRows; r++) {
			for (c=0; c<this.myCols; c++) {
				this.myData[r][c] = 0;
			}
		}
	}
	
	public int getRows() {
		return this.myRows;
	}
	
	public int getColumns() {
		return this.myCols;
	}
	
	public void changeMe(int row, int column, double myvalue) {
        
		this.myData[row][column] = myvalue;
	}
	
	public void updateAddMe(int row, int column, double myvalue) {
		
		this.myData[row][column] = this.myData[row][column] + myvalue;
	}
    
	public void updateSubMe(int row, int column, double myvalue) {
		
		this.myData[row][column] = this.myData[row][column] - myvalue;
	}
	
	public double getMe(int row, int column) {
		return this.myData[row][column];
	}
	
	public void printMe() {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable	
		
		System.out.print("[");
		for (r=0; r<this.myRows; r++) {
			System.out.print(" {");
			for (c=0; c<this.myCols; c++) {
				System.out.print(this.myData[r][c]);
				if(c<this.myRows-1) System.out.print(", ");
				else System.out.print("}");
			}
		}
		System.out.print(" ]\n");
	}
	
	public TestMatrix addMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		
		if (this.myRows==otherData.getRows() && this.myCols==otherData.getColumns()) {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<this.myCols;c++) {
					newData.changeMe(r, c, this.myData[r][c] + otherData.getMe(r, c));
				}
			}
		}
		
		else {
			System.out.println("Matrix addition error!  Matrices are not of the same size!");
			System.exit(1);
		}
		return newData;
	}
	
	public TestMatrix subMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix subtraction
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		
		if (this.myRows==otherData.getRows() && this.myCols==otherData.getColumns()) {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<this.myCols;c++) {
					newData.changeMe(r, c, this.myData[r][c] - otherData.getMe(r, c));
				}
			}
		}
		
		else {
			System.out.println("Matrix addition error!  Matrices are not of the same size! Returning an all-zero TestMatrix.");
		}
		return newData;
	}	
	
	public TestMatrix subMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		for(r=0;r<this.myRows;r++) {
			for(c=0;c<this.myCols;c++) {
				for(k=0; k<this.myCols; k++) {
					newData.updateSubMe(r,c, x);
				}					
			}
		}
		return newData;		
	}
    
	public TestMatrix mulMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		for(r=0;r<this.myRows;r++) {
			for(c=0;c<this.myCols;c++) {
				for(k=0; k<this.myCols; k++) {
					newData.changeMe(r,c, x);
				}					
			}
		}
		return newData;		
	}
	
	public TestMatrix mulMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, otherData.getColumns());
		if (this.myCols==otherData.getRows()) {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<otherData.getColumns();c++) {
					for(k=0; k<this.myCols; k++) {
						newData.updateAddMe(r,c, this.myData[r][k]*otherData.getMe(k, c));
					}					
				}
			}
		}
		
		else {
			System.out.println("Matrix multiplication error!  The number of columns in the first matrix must match the number of rows in the second matrix! Returning an all-zero TestMatrix.");
		}
		
		return newData;
	}
	
	public TestMatrix divMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix division
		int r;					// Row loop counting variable
		int c; 					// Column loop counting variable
		int k;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		for (r=0; r<this.myRows;r++) {
			for (c=0; c<this.myCols;c++) {
				for(k=0; k<this.myCols; k++) {
					newData.updateAddMe(r,c, this.myData[r][k]/x);
				}
			}	
		}
		return newData;
	}
	
	private void swapMe(int r1, int c1, int r2, int c2) {
		
		//Variable Declaration
		double p;				// double value placeholder
        
		p = this.myData[r1][c1];
		this.myData[r1][c1] = this.myData[r2][c2];
		this.myData[r2][c2] = p;
	}
	
	public TestMatrix copyMe() {
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		for(r=0; r< this.myRows; r++) {
			for(c=0; c<this.myCols; c++) {
				newData.changeMe(r,c,this.getMe(r,c));
			}
		}
		return newData;
	}
    
	public TestMatrix invMatrix() {
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		TestMatrix currentData;	// Placeholder for the current matrix
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int i;					// Loop counting variable
		int m;					// Placeholder column varialbe
		double p;				// double value placeholder
		double q;				// double value placeholder
		
		currentData = this.copyMe();
		newData = new TestMatrix(this.myRows, this.myCols);	
		if(this.myCols == this.myRows) {
			for(r=0; r<this.myCols; r++) newData.changeMe(r, r, 1);
			for(c=0; c<this.myCols; c++) {
				m=c;
				p=currentData.getMe(c, c);
				for(i=c+1; i<this.myRows; i++) {
					if (Math.abs(currentData.getMe(i, c))>Math.abs(p)) {
						m=i;
						p=currentData.getMe(i, c);
					}
				}
				
				for(i=0; i< this.myCols; i++) {
					currentData.swapMe(m, i, c, i);
					newData.swapMe(m,i,c,i);
				}
				for(i=0; i<this.myCols; i++) {
					currentData.changeMe(c, i, currentData.getMe(c, i)/p);
					newData.changeMe(c, i, newData.getMe(c, i)/p);
				}
				for(r=0; r<this.myRows; r++) {
					if(r!=c) {
						q = currentData.getMe(r, c);
						for(i = 0; i < this.myCols; i++) {
							currentData.updateSubMe(r,i, currentData.getMe(c, i)*q);
							newData.updateSubMe(r, i, newData.getMe(c, i)*q);
						}
					}
				}
			}
		}
		else {
			System.out.println("Columns and rows do not match.  Inversion not possible! Returning an all-zero TestMatrix.");
		}
		return newData;
	}
	
	/*
	 * Constructor a diagonal matrix
	 *  Parameters
	 *   - rows: the integer number of rows (also number of columns)
	 *   - fill: the value to be used to fill the matrix
	 *   - one: the value in the diagonal
	 */
	public TestMatrix identity() {
		
		// Variable Declaration
		int i; 		   // Loop counting variable
		TestMatrix M;  // Identity Matrix
		
		M = new TestMatrix(this.myRows,this.myRows);
		for(i=0; i<this.myRows; i++) M.changeMe(i,i,1.0f);
		return M;
	}
	
	public TestMatrix sqrtTM() {
		
		// Variable Declaration
		TestMatrix square; 	// The squared result matrix
		int r;				// Row loop counting variable
		int c;				// Column loop counting variable
		
		square = new TestMatrix(this.myRows, this.myCols);
		for(r=0; r<this.myRows; r++) {
			for(c=0; c<this.myCols; c++) {
				square.changeMe(r,c, myData[r][c]*myData[r][c]);
			}
		}
		return square;
	}
}




