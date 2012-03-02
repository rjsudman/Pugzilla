/*
 * LinearAlgebra.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */

/* 
 * LinearAlgebra is a library of linear algebra algorithms.
 */
public class LinearAlgebra {
	private static float ap = 0.000001f; // Default absolute precision
	private static float rp = 0.0001f;	 // Default relative precision
	private static int ns = 40;			 // Default number of steps
	private static int p = 1;
	
	public boolean is_almost_symmetric(TestMatrix x) {
		
		// Variable declaration
		float delta;	// Test for symmetric
		int r;			// Row loop counting variable
		int c;			// Column loop counting variable
		
		if( x.getColumns() != x.getRows()) return false;
		for(r=0; r<x.getRows()-1; r++) {
			for(c=0; c<x.getColumns()-1; c++) {
				delta = Math.abs(x.getMe(r, c)-x.getMe(c,r));
				if (delta > ap && delta>Math.max(Math.abs(x.getMe(r,c)), Math.abs(x.getMe(c,r)))*rp) return false;
			}
		}
		return true;
	}
	
	public boolean is_almost_zero(TestMatrix A) {
		
		// Variable declaration
		float delta;	// Test for zero
		int r;			// Row loop counting variable
		int c;			// Column loop counting variable
		
		for(r=0; r<A.getRows(); r++) {
			for(c=0; c<A.getColumns(); c++) {
				delta = Math.abs(A.getMe(r, c)-A.getMe(c, r));
				if(delta>ap && delta>Math.max(Math.abs(A.getMe(r, c)), Math.abs(A.getMe(c,r)))*rp) return false;
			}
		}
		return true;
	}
	
	public float norm(float A) {
		return Math.abs(A);
	}
	public float norm(TestMatrix A) {
		// Variable declaration
		int r;
		int c;
		float myNorm;
		
		myNorm = 0f;
		if(A.getRows()==1 || A.getColumns()==1){
			for(r=0; r<A.getRows(); r++) {
				for(c=0; c<Math.pow(A.getColumns(),p);c++) {
					myNorm+=norm(A.getMe(r,c));
				}
			}
			return (float) Math.pow(myNorm,p);
		}
		else if(p==1) {
			for(r=0; r<A.getRows(); r++) {
				for(c=0; c<A.getColumns(); c++) {
					myNorm+=norm(A.getMe(r,c));
				}
			}
			return (float) myNorm;
		}
		else {
			System.out.println("Norm not implemented for your case. Returning a norm of zero.");
		}
		return 0f;
	}
	
    //BROKEN
	public TestFunction condition_number(TestFunction f) {
		return f;
	}
	
	public TestMatrix exp(TestMatrix x) {
		
		// Variable Declaration
		TestMatrix t;
		TestMatrix s;
		int k;
		
		t = s = x.identity();
		for(k=1; k<ns; k++){
			t = t.mulMatrix(x.divMatrix((float)k));
			s = s.addMatrix(t);
			if(norm(t)<Math.max(ap,norm(s)*rp)) return s;
		}
		System.out.println("exp does not converge. Returning zero matrix.");
		return new TestMatrix(x.getRows(), x.getColumns());
	}
	
	
}