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
	private static float ap = 0.000001f;
	private static float rp = 0.0001f;
	private static int ns = 40;
	private static int p = 1;
	
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