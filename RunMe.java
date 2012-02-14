//***************************************************************************\\
// RunMe.java     															 \\
// BSD License																 \\
// Original Python code composed by Massimo Di Pierro with BSD licensing	 \\
// # Created by Massimo Di Pierro - BSD License                              \\
//***************************************************************************\\

public class RunMe {
	
	public static void main (String[] args) {
		TestMatrix A = new TestMatrix(3,3);
		TestMatrix B;
		TestMatrix C;
		A.changeMe(0,0,1); 
		A.changeMe(0,1,2); 
		A.changeMe(0,2,3);
		A.changeMe(1,0,1); 
		A.changeMe(1,1,0); 
		A.changeMe(1,2,3);
		A.changeMe(2,0,2); 
		A.changeMe(2,1,2); 
		A.changeMe(2,2,4); 
		B = A.invMatrix();
		C = A.mulMatrix(B);
		A.printMe();
		B.printMe();
		C.printMe();
	}
}



