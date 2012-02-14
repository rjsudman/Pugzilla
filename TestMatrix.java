//***************************************************************************\\
// TestMatrix.java															 \\
// BSD License																 \\
// Original Python code composed by Massimo Di Pierro with BSD licensing	 \\
// # Created by Massimo Di Pierro - BSD License                              \\
//***************************************************************************\\

import java.lang.reflect.Array;
import java.util.Arrays;
import java.lang.Math;

public class TestMatrix {

	// Global Variables
	private int myRows;
	private int myCols;
	private float[][] myData;
	
	// TestMatrix Constructor
	public TestMatrix(int rows, int columns) {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable
		
		myRows = rows;
		myCols= columns;
		myData = new float[myRows][myCols];
		for (r=0; r<myRows; r++) {
			for (c=0; c<myCols; c++) {
				myData[r][c] = 0;
			}
		}
	}
	
	public int getRows() {
		return myRows;
	}
	
	public int getColumns() {
		return myCols;
	}
	
	public void changeMe(int row, int column, float myvalue) {
	
		myData[row][column] = myvalue;
	}
	
	public void updateAddMe(int row, int column, float myvalue) {
		
		myData[row][column] = myData[row][column] + myvalue;
	}

	public void updateSubMe(int row, int column, float myvalue) {
		
		myData[row][column] = myData[row][column] - myvalue;
	}
	
	public float getMe(int row, int column) {
		return myData[row][column];
	}
	
	public void printMe() {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable	
		
		System.out.print("[");
		for (r=0; r<myRows; r++) {
			System.out.print(" {");
			for (c=0; c<myCols; c++) {
				System.out.print(myData[r][c]);
				if(c<myRows-1) System.out.print(", ");
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
		
		newData = new TestMatrix(myRows, myCols);
		
		if (myRows==otherData.getRows() && myCols==otherData.getColumns()) {
			for(r=0;r<myRows;r++) {
				for(c=0;c<myCols;c++) {
					newData.changeMe(r, c, myData[r][c] + otherData.getMe(r, c));
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
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(myRows, myCols);
		
		if (myRows==otherData.getRows() && myCols==otherData.getColumns()) {
			for(r=0;r<myRows;r++) {
				for(c=0;c<myCols;c++) {
					newData.changeMe(r, c, myData[r][c] - otherData.getMe(r, c));
				}
			}
		}
		
		else {
			System.out.println("Matrix addition error!  Matrices are not of the same size! Returning an all-zero TestMatrix.");
		}
		return newData;
	}	
	
	public TestMatrix mulMatrix(float x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(myRows, myCols);
		for(r=0;r<myRows;r++) {
			for(c=0;c<myCols;c++) {
				for(k=0; k<myCols; k++) {
					newData.updateAddMe(r,c, myData[r][k]*x);
				}					
			}
		}
		return newData;		
	}
	
	public TestMatrix mulMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(myRows, myCols);
		
		if (myCols==otherData.getRows()) {
			for(r=0;r<myRows;r++) {
				for(c=0;c<otherData.getColumns();c++) {
					for(k=0; k<myCols; k++) {
						newData.updateAddMe(r,c, myData[r][k]*otherData.getMe(k, c));
					}					
				}
			}
		}
		
		else {
			System.out.println("Matrix multiplication error!  The number of columns in the first matrix must match the number of rows in the second matrix! Returning an all-zero TestMatrix.");
		}
		
		return newData;
	}
	
	private void swapMe(int r1, int c1, int r2, int c2) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		float p;				// Float value placeholder

		p = myData[r1][c1];
		myData[r1][c1] = myData[r2][c2];
		myData[r2][c2] = p;
	}
	
	public TestMatrix copyMe() {
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(myRows, myCols);
		for(r=0; r< myRows; r++) {
			for(c=0; c<myCols; c++) {
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
		float p;				// Float value placeholder
		float q;				// Float value placeholder
		
		currentData = this.copyMe();
		newData = new TestMatrix(myRows, myCols);	
		
		if(myCols == myRows) {
			for(r=0; r<myCols; r++) newData.changeMe(r, r, 1);
			for(c=0; c<myCols; c++) {
				m=c;
				p=currentData.getMe(c, c);
				for(i=c+1; i<myRows; i++) {
					if (Math.abs(currentData.getMe(i, c))>Math.abs(p)) {
						m=i;
						p=currentData.getMe(i, c);
					}
				}
				for(i=0; i< myCols; i++) {
					currentData.swapMe(m, i, c, i);
					newData.swapMe(m,i,c,i);
				}
				for(i=0; i<myCols; i++) {
					currentData.changeMe(c, i, currentData.getMe(c, i)/p);
					newData.changeMe(c, i, newData.getMe(c, i)/p);
				}
				for(r=0; r<myRows; r++) {
					if(r!=c) {
						q = currentData.getMe(r, c);
						for(i = 0; i < myCols; i++) {
							currentData.updateSubMe(r,i, currentData.getMe(r, i)*q);
							newData.updateSubMe(r, i, newData.getMe(r, i)*q);
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
}




