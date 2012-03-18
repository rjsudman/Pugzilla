/*
 * TestMatrix.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */		

import java.lang.Math;

/** 
 * An implementation of basic matrix math which can be used with a library of 
 * linear algebra algorithms originally created in Python by Massimo Di Pierro 
 * and ported to Java.  All code released under BSD licensing.
 * 
 * @author					Ruthann Sudman
 * @version					0.1
 * @see LinearAlgebra
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class TestMatrix {
    
	// Global Variables
	private int myRows;				// The rows in the TestMatrix matrix.
	private int myCols;				// The columns in the TestMatrix matrix.
	private double [][] myData;		// The TestMatrix.
	
	/**
	 * TestMatrix constructor, initializing the original matrix to all 0.
	 * 
	 * @param rows 		Number of rows in the TestMatrix.
	 * @param columns	Number of Columns in the TestMatrix.
	 * @exception ArrayIndexOutOfBoundsException TestMatrix need to have at least 1 row and 1 column.
	 * @exception ArithmeticException No known exceptions.
	 */
	public TestMatrix(int rows, int columns) {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable
		
		try {
			this.myRows = rows;
			this.myCols= columns;
			this.myData = new double[this.myRows][this.myCols];
			for (r=0; r<this.myRows; r++) {
				for (c=0; c<this.myCols; c++) {
					this.myData[r][c] = 0;
				}
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **TestMatrix Constructor**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **TestMatrix Constructor**!" + f.getMessage());
		}
	}
	
	/**
	 * Get method that returns the number of rows in the TestMatrix.
	 * 
	 * @exception ArithmeticException No known exceptions.
	 * @return Number of rows in the TestMatrix myRows.
	 */
	public int getRows() {
		
		try {
			return this.myRows;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **getRows**!" + e.getMessage());
			return 0;
		}
	}
	
	/** 
	 * Get method that returns the number of columns in the TestMatrix.
	 * 
	 * @exception ArithmeticException No known exceptions.
	 * @return The number of columns in the TestMatrix myCols.
	 */
	public int getColumns() {
		
		try {
			return this.myCols;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **getCols**!" + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * Updates a specific value in the myData.
	 * 
	 * @param row		The row of the value to update.
	 * @param column	The column of the value to update.
	 * @param myvalue	The update value.
	 * @exception ArrayIndexOutOfBoundsException row and column must be in the bounds of the matrix myData.
	 * @exception ArithmeticException No known exceptions.
	 */
	public void changeMe(int row, int column, double myvalue) {
        
		try {
			this.myData[row][column] = myvalue;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **changeMe**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **changeMe**!" + f.getMessage());
		}
	}
	
	/**
	 * Adds a value to a specific value in the myData.
	 * 
	 * @param row		The row of the value to add to.
	 * @param column	The column of the value to add to.
	 * @param myvalue	The value to add to the original number.
	 * @exception ArrayIndexOutOfBoudsException row and column must be in the bounds of the matrix myData.
	 * @exception ArithmeticException No known exceptions.
	 */
	private void updateAddMe(int row, int column, double myvalue) {
		
		try {
			this.myData[row][column] = this.myData[row][column] + myvalue;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **updateAddMe**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **updateAddMe**!" + f.getMessage());
		}
	}
    
	/**
	 * Subtracts a specific value in the myData.
	 * 
	 * @param row		The row of the value to subtract from.
	 * @param column	The column of the value to subtract.
	 * @param myvalue	The value to subtract from the original number.
	 * @exception ArrayIndexOutOfBoundsException row and column must be in the bounds of the matrix myData.
	 * @exception ArithmeticException No known exceptions.
	 */
	public void updateSubMe(int row, int column, double myvalue) {
		try {
			this.myData[row][column] = this.myData[row][column] - myvalue;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **updateSubMe**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **updateSubMe**!" + f.getMessage());
		}
	}
	
	/**
	 * Obtain a specific value in the myData.
	 * 
	 * @param row		The row of the desired value.
	 * @param column	The column of the desired value.
	 * @exception ArrayIndexOutOfBoundsException row and column must be in the bounds of the matrix myData.
	 * @exception ArithmeticException No known exceptions
	 * @return	The desired value to return from myData.
	 */
	public double getMe(int row, int column) {
		
		try {
			return this.myData[row][column];
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **getMe**!" + e.getMessage());
			return 0;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **getMe**!" + f.getMessage());
			return 0;
		}
	}
	
	/**
	 * Print out the myData in a single line.
	 * 
	 * @exception ArrayIndexOutOfBoundsException row and column must be in the bounds of the matrix myData.
	 * @exception ArithmeticException No known exceptions.
	 * @exception Other Printing does not work well for TestMatrices with column = 1.
	 */
	public void printMe() {
		
		// Variable Declaration
		int r;		// Row loop counting variable
		int c;		// Column loop counting variable	
		
		try {
			System.out.print("[");
			for (r=0; r<this.myRows; r++) {
				System.out.print(" [");
				for (c=0; c<this.myCols; c++) {
					if(this.myRows == 1 && c>0) System.out.print(" [");
					System.out.print(this.myData[r][c]);
					if(c<this.myRows-1) System.out.print(", ");
					else System.out.print("]");
				}
			}
			System.out.print(" ]\n");
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **printMe**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **printMe**!" + f.getMessage());
		}
	}
	
	/**
	 * Add two TestMatrices together.
	 * 
	 * @param otherData	The TestMatrix to add to myData.
	 * @exception ArrayIndexOutOfBoundsException The rows and columns of both matrices must be equal.
	 * @exception ArithmeticException No known exceptions.
	 * @return The added TestMatrx.
	 */
	public TestMatrix addMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		try {
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
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **addMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **addMatrix**!" + f.getMessage());
			return newData;
		}
	}
	
	/**
	 * Add a value to all elements in the TestMatrix.
	 * 
	 * @param x	The value to add to all elements of the TestMatrix.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions
	 * @return The TestMatrix with the addition of x performed.
	 */
	public TestMatrix addMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		
		newData = this.copyMe();
		try {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<this.myCols;c++) {
					newData.updateAddMe(r,c, x);
				}
			}
			return newData;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **addMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **addMatrix**!" + f.getMessage());
			return newData;
		}
	}
	
	/**
	 * Subtract one TestMatrix from another.
	 * 
	 * @param otherData	The matrix to subtract from myData.
	 * @exception ArrayIndexOutOfBoundsException The rows and columns of both matrices must be equal.
	 * @exception ArithmeticException No known exceptions
	 * @return The subtracted TestMatrix.
	 */
	public TestMatrix subMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix subtraction
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		// Variable initialization;
		newData = this.copyMe();
		try {
			if (this.myRows==otherData.getRows() && this.myCols==otherData.getColumns()) {
				for(r=0;r<this.myRows;r++) {
					for(c=0;c<this.myCols;c++) {
						newData.updateSubMe(r, c, otherData.getMe(r, c));
					}
				}
			}
			
			else {
				System.out.println("Matrix addition error!  Matrices are not of the same size! Returning an all-zero TestMatrix.");
			}
			newData.printMe();
			return newData;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **subMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **subMatrix**!" + f.getMessage());
			return newData;
		}
	}	
	
	/**
	 * Subtract a specific value from all elements in the TestMatrix.
	 * 
	 * @param x The value to subtract from all elements in myData.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @return myData with x subtracted.
	 */
	public TestMatrix subMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		
		newData = this.copyMe();
		try {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<this.myCols;c++) {
						newData.updateSubMe(r,c, x);				
				}
			}
			return newData;		
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **subMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **subMatrix**!" + f.getMessage());
			return newData;
		}
	}
    
	/**
	 * Multiply all elements in a TestMatrix by a value.
	 * 
	 * @param x	The value to multiply all elements in myData by.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @return myData multiplied by x.
	 */
	public TestMatrix mulMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		
		newData = this.copyMe();
		try {
			for(r=0;r<this.myRows;r++) {
				for(c=0;c<this.myCols;c++) {
					newData.changeMe(r,c, x*newData.getMe(r,c));				
				}
			}
			return newData;		
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **mulMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **mulMatrix**!" + f.getMessage());
			return newData;
		}
	}
	
	/**
	 * Do a scalar multiplication of two matrices.
	 * 
	 * @param B The TestMatrix to be multiplied with myData.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @exception Other The number of rows for both TestMatrices must be one and the number of columns must be equal OR the number of columns for both TestMatrices must be one and the number of rows must be equal.
	 * @return The scalar multiplication of myData and TestMatrix B.
	 */
	public double mulMatrixScalar(TestMatrix B) {
		
		// Variable Declaration
		int r;			// Row loop counting variable
		double mySum;	// Scalar product of two matrices with 1 column and equal rows
		
		try {
			mySum=0;
			if(this.myCols == 1 && B.getColumns() == 1 && this.myRows == B.getRows()) {
				for (r=0; r<this.myRows; r++) {
					mySum+= this.myData[r][0]*B.getMe(r, 0);	
				}
				return mySum;
			}
			else if(this.myRows == 1 && B.getRows() == 1 && this.myCols == B.getColumns()) {
				for (r=0; r<this.myCols; r++) {
					mySum+= this.myData[0][r]*B.getMe(0, r);	
				}
				return mySum;
			}
			System.out.println("Arithmetic Error! **mulMatrix** expecting scalars with equal rows with a single column. Returning 0.");
			return 0;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in ****!" + e.getMessage());
			return 0;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in ****!" + f.getMessage());
			return 0;
		}
	}
	
	/**
	 * Multiply two TestMatrices together.
	 * 
	 * @param otherData	The TestMatrix to multiply with myData.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @exception Other The number of columns in myData must be equal to the number of rows in otherData.
	 * @return The multiplication of myData and TestMatrix otherData.
	 */
	public TestMatrix mulMatrix(TestMatrix otherData) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix multiplication
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable	
		int k; 					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, otherData.getColumns());
		try {
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
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **mulMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **mulMatrix**!" + f.getMessage());
			return newData;
		}
	}
	
	/**
	 * Divide all elements in a TestMatrix by x.
	 * 
	 * @param x	The value to divide all myData elements by.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @return The muliplication of myData and x.
	 */
	public TestMatrix divMatrix(double x) {
		
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix division
		int r;					// Row loop counting variable
		int c; 					// Column loop counting variable
		
		newData = this.copyMe();
		try {
			for (r=0; r<this.myRows;r++) {
				for (c=0; c<this.myCols;c++) {
					newData.changeMe(r,c, this.myData[r][c]/x);
				}	
			}
			return newData;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **divMatrix**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **divMatrix**!" + f.getMessage());
			return newData;
		}
	}
	
	/**
	 * Swap two values in myData.
	 * 
	 * @param r1	The row of the first value to swap.
	 * @param c1	The column of the first value to swap.
	 * @param r2	The row of the second value to swap.
	 * @param c2	The column of the second value to swap.
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 */
	private void swapMe(int r1, int c1, int r2, int c2) {
		
		//Variable Declaration
		double p;				// double value placeholder
        
		try {
			p = this.myData[r1][c1];
			this.myData[r1][c1] = this.myData[r2][c2];
			this.myData[r2][c2] = p;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **swapMe**!" + e.getMessage());
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **swapMe**!" + f.getMessage());
		}
	}
	
	/**
	 * Return a copy of myData. One cannot simply return myData 
	 * because that would be returning a double array and not a TestMatrix object.
	 * 
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @return TestMatrix
	 */
	public TestMatrix copyMe() {
		//Variable Declaration
		TestMatrix newData;		// Placeholder for the matrix addition
		int r;					// Row loop counting variable
		int c;					// Column loop counting variable
		
		newData = new TestMatrix(this.myRows, this.myCols);
		try {
			for(r=0; r< this.myRows; r++) {
				for(c=0; c<this.myCols; c++) {
					newData.changeMe(r,c,this.getMe(r,c));
				}
			}
			return newData;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **copyMe**!" + e.getMessage());
			return newData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **copyMe**!" + f.getMessage());
			return newData;
		}
	}
    
	/**
	 * The inverse of a TestMatrix object.
	 * 
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @return The inverse of myData.
	 */
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
		try {
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
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **invMatrix**!" + e.getMessage());
			return currentData;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in **invMatrix**!" + f.getMessage());
			return currentData;
		}
	}
	
	/**
	 * Construct a diagonal matrix identical in size to myData.
	 * 
	 * @exception ArrayIndexOutOfBoundsException Improper index value used for array.
	 * @exception ArithmeticException No known exceptions.
	 * @exception Other The size of the identity matrix will be identical to myData. Size is not customizable.
	 * @return The identity matrix for myData
	 */
	public TestMatrix identity() {
		
		// Variable Declaration
		int i; 		   // Loop counting variable
		TestMatrix M;  // Identity Matrix
		
		M = new TestMatrix(this.myRows,this.myRows);
		try {
			for(i=0; i<this.myRows; i++) M.changeMe(i,i,1.0f);
			return M;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in ****!" + e.getMessage());
			return M;
		}
		catch (ArrayIndexOutOfBoundsException f) {
			System.err.println("Array index out of bounds in ****!" + f.getMessage());
			return M;
		}
	}
	
	/**
	 * Return the norm of myData
	 * 
	 * @exception Other This function is not properly implemented.
	 * @return Norm of matrix myData
	 */
	public double norm() {
		/* We will assume p to be 1 */
		// Variable declaration
		//int r;			// Row counting variable
		//int c;			// Column counting variable
		//double myNorm;	// The norm of the TestMatrix
		
		System.err.println("**norm** not implemented for type TestMatrix. Returning zero.");
		return 0;
	}
		
	/**
	 * Return the condition number of myData.
	 * 
	 * @exception Other This function is not properly implemented.
	 * @return The condition number of myData.
	 */
	public double condition_number() {
		/* 
		 * 	def condition_number(f,x=None,h=1e-6):
		 * 		if callable(f) and not x is None:
		 *      	return D(f,h)(x)*x/f(x)
		 *  	elif isinstance(f,Matrix): # if is the Matrix Jƒzz
		 *      	return norm(f)*norm(1/f)
		 *  	else:
		 *      	raise NotImplementedError
		 */
		
		System.err.println("**condition_number** not implemented for type TestMatrix. Returning zero.");
		return 0;
	}
}




