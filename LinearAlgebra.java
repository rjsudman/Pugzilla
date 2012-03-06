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
		float myNorm;	// The norm value
		
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
	public float condition_number(TestFunction f) {
		/* 
		 * 	def condition_number(f,x=None,h=1e-6):
		 * 		if callable(f) and not x is None:
		 *      	return D(f,h)(x)*x/f(x)
		 *  	elif isinstance(f,Matrix): # if is the Matrix Jƒzz
		 *      	return norm(f)*norm(1/f)
		 *  	else:
		 *      	raise NotImplementedError
		 */
		
		System.out.println("The condition number algorithm has not yet been implemented.");
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
			t = t.mulMatrix(x.divMatrix((float)k));
			s = s.addMatrix(t);
			if(norm(t)<Math.max(ap,norm(s)*rp)) return s;
		}
		System.out.println("exp does not converge. Returning zero matrix.");
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
		float p;
		
		if (! is_almost_symmetric(A)) {
			System.out.println("Arithmetic Error! Matrix is not symmetric. Cholesky will not be performed.");
		}
		L = A.copyMe();
		for(k=0; k<L.getColumns(); k++) {
			if (L.getMe(k, k)<=0) {
				System.out.println("Arithmetic Error! Not positive definitive. Cholesky will not be completed.");
				k = L.getColumns();
			}
			else {
				p= (float) Math.sqrt(L.getMe(k, k));
				L.changeMe(k, k, p);
				for(i=k+1; i<L.getRows(); i++) {
					L.changeMe(i, k, L.getMe(i, k)/p);
				}
				for(j=k+1;j<L.getRows();j++) {
					p= L.getMe(j, k);
					for(i=k+1; k<L.getRows(); k++) {
						L.changeMe(i, j, L.getMe(i,j)-L.getMe(i, k)*p);
					}
				}
			}
			for(i=0;i<L.getRows();i++) {
				for(j=i+1;j<L.getColumns();j++) {
					L.changeMe(i, j, 0);
				}
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
    
	public TestMatrix Markovitz(float mu, TestMatrix A, float r_free) {
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
		int r;
		TestMatrix x;
		float p;
		TestMatrix portfolio;
		TestMatrix portfolio_return;
		TestMatrix portfolio_risk;
		TestMatrix Mark;
		
		p = 0;
		Mark= new TestMatrix(3,3);
		x=new TestMatrix(A.getRows(),1);
		x = A.invMatrix();
		x= x.mulMatrix(mu-r_free);
		for(r=0;r<x.getRows();r++) {
			p += x.getMe(r, 0);
		}
		x.divMatrix(p);
		portfolio = new TestMatrix(A.getRows(),1);
		for(r=0; r<x.getRows();r++) {
			portfolio.changeMe(r, 0, x.getMe(r, 0));
		}
		portfolio_return = x.copyMe();
		portfolio_return=portfolio_return.mulMatrix(mu);
		portfolio_risk=A.mulMatrix(x);
		portfolio_risk=portfolio_risk.mulMatrix(x);
		portfolio_risk = portfolio_risk.sqrtTM();
		for(r=0;r<3;r++) {
			Mark.changeMe(r,0,portfolio.getMe(r,0));
			Mark.changeMe(r,1,portfolio_return.getMe(r, 0));
			Mark.changeMe(r,2,portfolio_risk.getMe(r,0));
		}
		return Mark;
	}
	
	public TestMatrix fit_least_squares(TestMatrix points, TestFunction f) {
		/*
		 *		def fit_least_squares(points, f):
		 * 			"""
		 * 		    Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
		 * 		    where fitting_f(x[i]) is \sum_j c_j f[j](x[i])
		 * 
		 * 	    parameters:
		 * 		    - a list of fitting functions
		 * 		    - a list with points (x,y,dy)
		 * 
		 * 	    returns:
		 * 		    - column vector with fitting coefficients
		 * 		    - the chi2 for the fit
		 * 		    - the fitting function as a lambda x: ....
		 * 		    """
		 *	    def eval_fitting_function(f,c,x):
		 *	        if len(f)==1: return c*f[0](x)
		 *	        else: return sum(func(x)*c[i,0] for i,func in enumerate(f))
		 *		    A = Matrix(len(points),len(f))
		 *		    b = Matrix(len(points))
		 *	    for i in range(A.rows):
		 *	        weight = 1.0/points[i][2] if len(points[i])>2 else 1.0
		 *        	b[i,0] = weight*float(points[i][1])
		 *          for j in range(A.cols):
		 *              A[i,j] = weight*f[j](float(points[i][0]))
		 *        	c = (1.0/(A.t*A))*(A.t*b)
		 *          chi = A*c-b
		 *          chi2 = norm(chi,2)**2
		 *          fitting_f = lambda x, c=c, f=f, q=eval_fitting_function: q(f,c,x)
		 *          return c.data, chi2, fitting_f
		 */
		
		// Variable Declaration
		return points;
	}
	/*
     def solve_fixed_point(f, x, ap=1e-6, rp=1e-4, ns=100):
     def g(x): return f(x)+x # f(x)=0 <=> g(x)=x
     Dg = D(g)
     for k in xrange(ns):
     if abs(Dg(x)) >= 1:
     raise ArithmeticError, 'error D(g)(x)>=1'
     (x_old, x) = (x, g(x))
     if k>2 and norm(x_old-x)<max(ap,norm(x)*rp):
     return x
     raise ArithmeticError, 'no convergence'
     
     def solve_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     fa, fb = f(a), f(b)
     if fa == 0: return a
     if fb == 0: return b
     if fa*fb > 0:
     raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
     for k in xrange(ns):
     x = (a+b)/2
     fx = f(x)
     if fx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
     elif fx * fa < 0: (b,fb) = (x, fx)
     else: (a,fa) = (x, fx)
     raise ArithmeticError, 'no convergence'
     
     
     
     def solve_secant(f, x, ap=1e-6, rp=1e-4, ns=20):
     x = float(x) # make sure it is not int
     (fx, Dfx) = (f(x), D(f)(x))
     for k in xrange(ns):
     if norm(Dfx) < ap:
     raise ArithmeticError, 'unstable solution'
     (x_old, fx_old,x) = (x, fx, x-fx/Dfx)
     if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
     fx = f(x)
     Dfx = (fx-fx_old)/(x-x_old)
     raise ArithmeticError, 'no convergence'
     
     def solve_newton_stabilized(f, a, b, ap=1e-6, rp=1e-4, ns=20):
     fa, fb = f(a), f(b)
     if fa == 0: return a
     if fb == 0: return b
     if fa*fb > 0:
     raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
     x = (a+b)/2
     (fx, Dfx) = (f(x), D(f)(x))
     for k in xrange(ns):
     x_old, fx_old = x, fx
     if norm(Dfx)>ap: x = x - fx/Dfx
     if x==x_old or x<a or x>b: x = (a+b)/2
     fx = f(x)
     if fx==0 or norm(x-x_old)<max(ap,norm(x)*rp): return x
     Dfx = (fx-fx_old)/(x-x_old)
     if fx * fa < 0: (b,fb) = (x, fx)
     else: (a,fa) = (x, fx)
     raise ArithmeticError, 'no convergence'
     
     def optimize_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     Dfa, Dfb = D(f)(a), D(f)(b)
     if Dfa == 0: return a
     if Dfb == 0: return b
     if Dfa*Dfb > 0:
     raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
     for k in xrange(ns):
     x = (a+b)/2
     Dfx = D(f)(x)
     if Dfx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
     elif Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
     else: (a,Dfa) = (x, Dfx)
     raise ArithmeticError, 'no convergence'
     
     
     
     def optimize_secant(f, x, ap=1e-6, rp=1e-4, ns=100):
     x = float(x) # make sure it is not int
     (fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
     for k in xrange(ns):
     if Dfx==0: return x
     if norm(DDfx) < ap:
     raise ArithmeticError, 'unstable solution'
     (x_old, Dfx_old, x) = (x, Dfx, x-Dfx/DDfx)
     if norm(x-x_old)<max(ap,norm(x)*rp): return x
     fx = f(x)
     Dfx = D(f)(x)
     DDfx = (Dfx - Dfx_old)/(x-x_old)
     raise ArithmeticError, 'no convergence'
     
     def optimize_newton_stabilized(f, a, b, ap=1e-6, rp=1e-4, ns=20):
     Dfa, Dfb = D(f)(a), D(f)(b)
     if Dfa == 0: return a
     if Dfb == 0: return b
     if Dfa*Dfb > 0:
     raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
     x = (a+b)/2
     (fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
     for k in xrange(ns):
     if Dfx==0: return x
     x_old, fx_old, Dfx_old = x, fx, Dfx
     if norm(DDfx)>ap: x = x - Dfx/DDfx
     if x==x_old or x<a or x>b: x = (a+b)/2
     if norm(x-x_old)<max(ap,norm(x)*rp): return x
     fx = f(x)
     Dfx = (fx-fx_old)/(x-x_old)
     DDfx = (Dfx-Dfx_old)/(x-x_old)
     if Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
     else: (a,Dfa) = (x, Dfx)
     raise ArithmeticError, 'no convergence'
     
     def optimize_golden_search(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     a,b=float(a),float(b)
     tau = (sqrt(5.0)-1.0)/2.0
     x1, x2 = a+(1.0-tau)*(b-a), a+tau*(b-a)
     fa, f1, f2, fb = f(a), f(x1), f(x2), f(b)
     for k in xrange(ns):
     if f1 > f2:
     a, fa, x1, f1 = x1, f1, x2, f2
     x2 = a+tau*(b-a)
     f2 = f(x2)
     else:
     b, fb, x2, f2 = x2, f2, x1, f1
     x1 = a+(1.0-tau)*(b-a)
     f1 = f(x1)
     if k>2 and norm(b-a)<max(ap,norm(b)*rp): return b
     raise ArithmeticError, 'no convergence'
     
	 */
	
	
}