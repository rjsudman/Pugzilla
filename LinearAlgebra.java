/*
 * LinearAlgebra.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */

/** 
 * A library of linear algebra algorithms originally created in Python by 
 * Massimo Di Pierro and ported to Java.   All code released under BSD licensing.
 * 
 * @author					Ruthann Sudman
 * @version					0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public class LinearAlgebra {
	private static double ap = 0.000001; // Default absolute precision
	private static double rp = 0.0001;	 // Default relative precision
	private static int ns = 40;			 // Default number of steps
	private static int p = 1;
	private TestMatrix portfolio;				 // Markovitz portfolio
	private double portfolio_return;			 // Markovitz return
	private double portfolio_risk;				 // Markovitz risk
	private boolean MarkovitzStatus;			 // Has Markovitz been run?
    
	/** 
	 * Constructor
	 */
	public LinearAlgebra () {
		
		// Initializing status of Markovitz to 0.
		MarkovitzStatus = false;
	}
	/**
	 * Returns a boolean value indicating whether the matrix is almost
	 * symmetric.
	 * 
	 * @param x		The TestMatrix object to be examined.
	 * @return		The boolean result of the test.
	 * @exception	ArithmeticException No known exceptions.
	 * @see			TestMatrix	
	 */
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
		
		try {
			if( x.getColumns() != x.getRows()) return false;
			for(r=0; r<x.getRows()-1; r++) {
				for(c=0; c<x.getColumns()-1; c++) {
					delta = Math.abs(x.getMe(r, c)-x.getMe(c,r));
					if (delta > ap && delta>Math.max(Math.abs(x.getMe(r,c)), Math.abs(x.getMe(c,r)))*rp) return false;
				}
			}
			return true;
		}
		catch (ArithmeticException e){
			System.err.println("Arithmetic exception in **is_almost_symmetric**! " + e.getMessage());
			return false;
		}
	}
	
	/**
	 * Returns a boolean value indicating if a matrix is almost zero.
	 * 
	 * @param A		The TestMatrix object to be examined.
	 * @exception	ArithmeticException No known exceptions.
	 * @return		Boolean result of the test.
	 * @see			TestMatrix
	 */
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
		
		try {
			for(r=0; r<A.getRows(); r++) {
				for(c=0; c<A.getColumns(); c++) {
					delta = Math.abs(A.getMe(r, c)-A.getMe(c, r));
					if(delta>ap && delta>Math.max(Math.abs(A.getMe(r, c)), Math.abs(A.getMe(c,r)))*rp) return false;
				}
			}
			return true;			
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **is_almost_zero**! " + e.getMessage());
			return false;
		}
	}
	
	/**
	 * Returns the norm of a double value.
	 * @param 		A	The value to be examined.
	 * @exception	ArithmeticException No known exceptions.
	 * @return		The norm of A.
	 */
	public double norm(double A) {
		
		try { 
			return Math.abs(A);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **norm**! " + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * Returns the norm of a TestMatrix. Needs work. Not properly implemented.
	 * @param A		The TestMatrix object to be examined.
	 * @exception	ArithmeticException Norm will always be zero. Not properly implemented.
	 * @return		The norm of the matrix.
	 */
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
		
		try {
			myNorm = 0;
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
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **norm**! " + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **norm** not implemented for your case. Returning zero.");
		return 0;
	}
    
	/**
	 * Returns the exponent of a TestMatrix object.
	 * @param x		The TestMatrix object to apply the function to.
	 * @exception	ArithmeticException Algorithm may fail to converge, division by zero errors.
	 * @return		The exponent TestMatrix.
	 * @see 		TestMatrix
	 */
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
		
		try {
			t = s = x.identity();
			for(k=1; k<ns; k++){
				t = t.mulMatrix(x.divMatrix(k));
				s = s.addMatrix(t);
				if(norm(t)<Math.max(ap,norm(s)*rp)) return s;
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **exp**! " + e.getMessage());
			return new TestMatrix(x.getRows(), x.getColumns());			
		}
		System.err.println("Arithmetic Error! **exp** does not converge. Returning zero.");
		return new TestMatrix(x.getRows(), x.getColumns());
	}
	
	/**
	 * Returns a TestMatrix object with the Cholesky algorithm applied.
	 * @param A		The TestMatrix object to apply Cholesky to.
	 * @exception	ArithmeticException Can't take a square root of a negative number.
	 * @return		A TestMatrix with Cholesky applied.
	 * @see 		TestMatrix
	 */
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
			System.err.println("Arithmetic Error! Matrix is not symmetric for **Cholesky**. Returning zero.");
			L = new TestMatrix(A.getRows(), A.getColumns());
			return L;
		}
		L = A.copyMe();
		for(k=0; k<L.getColumns(); k++) {
			if (L.getMe(k, k)<=0) {
				System.err.println("Arithmetic Error! Not positive definitive for **Cholesky**. Returning zero.");
				L = new TestMatrix(A.getRows(), A.getColumns());
				return L;
			}
            try {
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
            catch (ArithmeticException e) {
            	System.err.println("Arithmetic exception in Cholesky! " + e.getMessage());
            	L = new TestMatrix(A.getRows(), A.getColumns());
				return L;
            }
		}
		try {
			for(i=0;i<L.getRows();i++) {
				for(j=i+1;j<L.getColumns();j++) {
					L.changeMe(i, j, 0);
				}
			}
			return L;			
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in Cholesky! " + e.getMessage());
        	L = new TestMatrix(A.getRows(), A.getColumns());
			return L;			
		}
	}
	
	/**
	 * Returns a boolean value indicating if a TestMatrix is positive definite.
	 * @param A		The TestMatrix to test for positive definite.
	 * @exception	ArithmeticException Run time error possible.
	 * @return		The boolean result of the algorithm.
	 * @see			TestMatrix
	 */
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
		
		try {
			myTest = true;
			if(! is_almost_symmetric(A)) myTest = false;
			for(k=0; k<A.getColumns();k++) {
				if(A.getMe(k,k)<=0) {
					myTest = false;
				}
			}
	        return myTest;			
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **is_positive_definite**! " + e.getMessage());
			return false;
		}
	}
    /**
     * Calculates the Markovitz portfolio, risk and return. Returns a reference 
     * to LinearAlgebra from which the Markovitz portfolio TestMatrix, risk and 
     * return can be obtained with get methods.
     * 
     * @param mu		Markovitz mu.
     * @param A			The TestMatrix object.
     * @param r_free	The risk free rate.
     * @return			LinearAlgebra reference to get portfolio, risk and return
     * @exception		ArithmeticException TestMatrix should be symmetric. Rows in mu should mirror columns in A
     * @see				LinearAlgebra#getMarkovitzPortfolio()
     * @see				LinearAlgebra#getMarkovitzPortfolioRisk()
     * @see				LinearAlgebra#getMarkovitzPortfolioReturn()
     */
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
		TestMatrix m;	// PlaceholderT TestMatrix for mu
		TestMatrix a;	// Placeholder TestMatrix for A
        
		try {
			x = new TestMatrix(A.getRows(),1);
			m = mu.subMatrix(r_free);
			a = A.invMatrix();
			x = a.mulMatrix(m);
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
			portfolio_risk = Math.sqrt(portfolio_risk);
			MarkovitzStatus = true;
			return this;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **Markovitz**! " + e.getMessage());
			MarkovitzStatus = false;
			return this;
		}
	}
	
	/**
	 * Get method to return Markovitz portfolio value.
	 * @return 		Portfolio TestMatrix object
	 * @exception	MarkovitzFirst Markovitz must be run for this value to make any sense.
	 * @see			#Markovitz(TestMatrix, TestMatrix, double)
	 * @see			TestMatrix
	 */
	public TestMatrix getMarkovitzPortfolio() {

		if (MarkovitzStatus) return this.portfolio;
		else {
			System.err.println("Markovitz must be called first. Returning 0 portfolio.);");
			return this.portfolio;
		}
	}
	
	/**
	 * Get method to return Markovitz portfolio risk.
	 * @exception	MarkovitzFirst Markovitz must be run for this value to make any sense.
	 * @return	Markovitz portfolio risk.
	 * @see		#Markovitz(TestMatrix, TestMatrix, double)
	 */
	public double getMarkovitzPortfolioRisk() {
		if (MarkovitzStatus) return this.portfolio_risk;
		else {
			System.err.println("Markovitz must be called first. Returning 0 risk.);");
			return 0;
		}
	}
	
	/**
	 * Get method to return Markovitz portfolio return.
	 * @exception	MarkovtizFirst Markovitz must be run for this value to make any sense.
	 * @return	Markovitz portfolio return.
	 * @see		#Markovitz(TestMatrix, TestMatrix, double)
	 */
	public double getMarkovitzPortfolioReturn() {
		if (MarkovitzStatus) return this.portfolio_return;
		else {
			System.err.println("Markovitz must be called first. Returning 0 return.);");
			return 0;
		}
	}
	
	/**
	 * Returns the square root of a double.
	 * 
	 * @param x	The number to take the square root of.
	 * @exception Does not work for negative numbers.
	 * @return The square root of x.
	 */
	public double MySqrt(double x) {
		
	    try {
	        return Math.sqrt(x);
	    }
	    catch (ArithmeticException e) {
	    	System.err.println("Arithmetic Exception in MySqrt: " + e.getMessage());
	    	return 0;
	    }
	    
	}
}