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
	private static double ap = 0.000001f; // Default absolute precision
	private static double rp = 0.0001f;	 // Default relative precision
	private static int ns = 40;			 // Default number of steps
	private static int p = 1;
	TestMatrix portfolio;				 // Markovitz portfolio
	double portfolio_return;			 // Markovitz return
	double portfolio_risk;				 // Markovitz risk
    
	
	public boolean is_almost_symmetric(TestMatrix x) {
		/*
		 * 	def is_almost_symmetric(A, ap=1e-6, rp=1e-4):
		 *  	if A.rows != A.cols: return False
		 *  	for r in xrange(A.rows-1):
		 *  	    for c in xrange(r):
		 *      	    delta = abs(A[r,c]-A[c,r])
		 *          	if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
		 * 	            	return False
		 *  	return True
		 */
		
		// Variable declaration
		double delta;	// Test for symmetric
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
		/*
		 * 	def is_almost_zero(A, ap=1e-6, rp=1e-4):
		 * 		for r in xrange(A.rows):
		 * 			for c in xrange(A.cols):
		 * 				delta = abs(A[r,c]-A[c,r])
		 *  			if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
		 *      			return False
		 *  	return True
		 */
		
		// Variable declaration
		double delta;	// Test for zero
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
	
	public double norm(double A) {
		return Math.abs(A);
	}
	public double norm(TestMatrix A) {
		/*
		 *	def norm(A,p=1):
		 * 		if isinstance(A,(list,tuple)):
		 * 			return sum(x**p for x in A)**(1.0/p)
		 * 		elif isinstance(A,Matrix):
		 * 			if A.rows==1 or A.cols==1:
		 *   			return sum(norm(A[r,c])**p \
		 *      			for r in xrange(A.rows) \
		 *      			for c in xrange(A.cols))**(1.0/p)
		 *      	elif p==1:
		 *      		return max([sum(norm(A[r,c]) \
		 *      			for r in xrange(A.rows)) \
		 *      			for c in xrange(A.cols)])
		 *  		else:
		 *   			raise NotImplementedError
		 * 		else:
		 * 			return abs(A)
		 */
		// Variable declaration
		int r;			// Row loop counting variable
		int c;			// Column loop counting variable
		double myNorm;	// The norm value
		
		myNorm = 0f;
		if(A.getRows()==1 || A.getColumns()==1){
			for(r=0; r<A.getRows(); r++) {
				for(c=0; c<Math.pow(A.getColumns(),p);c++) {
					myNorm+=norm(A.getMe(r,c));
				}
			}
			return Math.pow(myNorm,p);
		}
		else if(p==1) {
			for(r=0; r<A.getRows(); r++) {
				for(c=0; c<A.getColumns(); c++) {
					myNorm+=norm(A.getMe(r,c));
				}
			}
			return myNorm;
		}
		else {
			System.out.println("Arithmetic Error! **norm** not implemented for your case. Returning zero.");
		}
		return 0f;
	}
    
	public TestMatrix exp(TestMatrix x) {
		/*
		 * 	def exp(x,ap=1e-6,rp=1e-4,ns=40):
		 *  	if isinstance(x,Matrix):
		 *     		t = s = Matrix.identity(x.cols)
		 *     		for k in range(1,ns):
		 *         		t = t*x/k   # next term
		 *         		s = s + t   # add next term
		 *         		if norm(t)<max(ap,norm(s)*rp): return s
		 *    		raise ArithmeticError, 'no convergence'
		 *   	elif type(x)==type(1j):
		 *     		return cmath.exp(x)
		 *  	else:
		 *     		return math.exp(x)
		 */
		
		// Variable Declaration
		TestMatrix t;	// Identity matrix of x
		TestMatrix s;	// Identity matrix of x
		int k;			// Loop counting variable
		
		t = s = x.identity();
		for(k=1; k<ns; k++){
			t = t.mulMatrix(x.divMatrix(k));
			s = s.addMatrix(t);
			if(norm(t)<Math.max(ap,norm(s)*rp)) return s;
		}
		System.out.println("Arithmetic Error! **exp** does not converge. Returning zero.");
		return new TestMatrix(x.getRows(), x.getColumns());
	}
	
	public TestMatrix Cholesky(TestMatrix A) {
		/*
		 * 	def Cholesky(A):
		 * 		import copy, math
		 * 		if not is_almost_symmetric(A):
		 *  		raise ArithmeticError, 'not symmetric'
		 *  	L = copy.deepcopy(A)
		 *  	for k in xrange(L.cols):
		 *  		if L[k,k]<=0:
		 *      		raise ArithmeticError, 'not positive definitive'
		 *  		p = L[k,k] = math.sqrt(L[k,k])
		 *  		for i in xrange(k+1,L.rows):
		 *      		L[i,k] /= p
		 *  		for j in xrange(k+1,L.rows):
		 *      		p=float(L[j,k])
		 *      		for i in xrange(k+1,L.rows):
		 *          		L[i,j] -= p*L[i,k]
		 * 		for  i in xrange(L.rows):
		 *  		for j in range(i+1,L.cols):
		 *      		L[i,j]=0
		 * 	return L
		 */
		
		// Variable declaration
		TestMatrix L;	// Copy of matrix A
		int i;			// Row loop counting variable
		int j;			// Row loop counting variable
		int k; 			// Column loop counting variable
		double p;
		
		if (! is_almost_symmetric(A)) {
			System.out.println("Arithmetic Error! Matrix is not symmetric for **Cholesky**. Returning zero.");
			L = new TestMatrix(A.getRows(), A.getColumns());
			return L;
		}
		L = A.copyMe();
		for(k=0; k<L.getColumns(); k++) {
			if (L.getMe(k, k)<=0) {
				System.out.println("Arithmetic Error! Not positive definitive for **Cholesky**. Returning zero.");
				L = new TestMatrix(A.getRows(), A.getColumns());
				return L;
			}
            
			p= Math.sqrt(L.getMe(k, k));
			L.changeMe(k, k, p);
			for(i=k+1; i<L.getRows(); i++) {
				L.changeMe(i, k, L.getMe(i, k)/p);
			}
			for(j=k+1;j<L.getRows();j++) {
				p= L.getMe(j, k);
				for(i=k+1; i<L.getRows(); i++) {
					L.changeMe(i, j, L.getMe(i,j)-L.getMe(i, k)*p);
				}
			}
		}
		for(i=0;i<L.getRows();i++) {
			for(j=i+1;j<L.getColumns();j++) {
				L.changeMe(i, j, 0);
			}
		}
		return L;
	}
	
	public boolean is_positive_definite(TestMatrix A){
		/*
		 * 	def is_positive_definite(A):
		 *		if not is_symmetric(A):
		 *	      return False
		 *  try:
		 *      Cholesky(A)
		 *      return True
		 *  except RuntimeError:
		 *      return False
		 */
		
		// Variable declaration
		boolean myTest;		// Test for positive definitive
		int k;				// Loop counting variable
		
		myTest = true;
		if(! is_almost_symmetric(A)) myTest = false;
		for(k=0; k<A.getColumns();k++) {
			if(A.getMe(k,k)<=0) {
				myTest = false;
			}
		}
        return myTest;
	}
    
	public LinearAlgebra Markovitz(TestMatrix mu, TestMatrix A, double r_free) {
		/*
		 * 		def Markovitz(mu, A, r_free):
		 * 		    """Assess Markovitz risk/return.
		 * 		    Example:
		 * 			>>> cov = Matrix.from_list([[0.04, 0.006,0.02],
		 * 			...                        [0.006,0.09, 0.06],
		 * 		    ...                        [0.02, 0.06, 0.16]])
		 * 		    >>> mu = Matrix.from_list([[0.10],[0.12],[0.15]])
		 * 			>>> r_free = 0.05
		 * 		    >>> x, ret, risk = Markovitz(mu, cov, r_free)
		 * 			>>> print x
		 * 		    [0.556634..., 0.275080..., 0.1682847...]
		 * 		    >>> print ret, risk
		 * 		    0.113915... 0.186747...
		 * 		    """
		 * 		    x = Matrix(A.rows, 1)
		 * 		    x = (1/A)*(mu - r_free)
		 * 		    x = x/sum(x[r,0] for r in range(x.rows))
		 * 		    portfolio = [x[r,0] for r in range(x.rows)]
		 * 		    portfolio_return = mu*x
		 * 		    portfolio_risk = sqrt(x*(A*x))
		 * 		    return portfolio, portfolio_return, portfolio_risk
		 */
		
		// Variable declaration
		int r;			// Row loop counting variable
		double p;		// Math placeholder
		TestMatrix x;	// Placeholder TestMatrix
		TestMatrix a;	// Placeholder TestMatrix
        
		x = new TestMatrix(A.getRows(),1);
		mu = mu.subMatrix(r_free);
		a = A.invMatrix();
		x = a.mulMatrix(mu);
		p= 0;
		for(r=0; r<x.getRows(); r++) {
			p += x.getMe(r,0);
		}
		x= x.divMatrix(p);
		portfolio = new TestMatrix(1,x.getRows());
		for(r=0; r< x.getRows();r++) {
			portfolio.changeMe(0,r,x.getMe(r,0));
		}
		portfolio_return = mu.mulMatrixScalar(x);
		A = A.mulMatrix(x);
		portfolio_risk = A.mulMatrixScalar(x);
		return this;	
	}
	
	public TestMatrix getMarkovitzPortfolio() {
		return this.portfolio;
	}
	
	public double getMarkovitzPortfolioRisk() {
		return this.portfolio_risk;
	}
	
	public double getMarkovitzPortfolioReturn() {
		return this.portfolio_return;
	}
}