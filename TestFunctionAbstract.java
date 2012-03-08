/*
 * TestFunctionAbstract.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 */		

/* 
 * The TestFunction class implements the concept of a function.
 */
public abstract class TestFunctionAbstract {
    
	public static double h=0.00001;
	public static double ap=0.00001;
	public static double rp=0.0001;
	public static int ns=100;
	private static LinearAlgebra A = new LinearAlgebra();
	
	public abstract double f(double x);
	
	public double Df(double x) {
		return (f(x+h)-f(x-h))/(2.0*h);
	}
    
	public double DDf(double x) {
		return (f(x+h)-2.0*f(x)+f(x-h))/(h*h);
	}
	
	public double g(double x) {
		return f(x)+x;
	}
	public double Dg(double x) {
		return (f(x+h)-f(x-h))/(2.0*h)+x;
	}
    
	public double condition_number(TestFunction f, double x) {
		/* 
		 * 	def condition_number(f,x=None,h=1e-6):
		 * 		if callable(f) and not x is None:
		 *      	return D(f,h)(x)*x/f(x)
		 *  	elif isinstance(f,Matrix): # if is the Matrix Jƒzz
		 *      	return norm(f)*norm(1/f)
		 *  	else:
		 *      	raise NotImplementedError
		 */
		return Df(x)*x/f(x);
	}
	
	public double condition_number(TestMatrix f) {
		
		// Variable declaration
		TestMatrix conditionMe;		// The condition matrix
		double conditionMe2;		// The condition number
        
		conditionMe = f.invMatrix();
		conditionMe2 = A.norm(f)*A.norm(conditionMe);
		return conditionMe2;
	}
	
	// BROKEN
	public TestMatrix fit_least_squares(double x, double y, double dy) {
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
		TestMatrix points = new TestMatrix(3,1);
		
		System.out.println("fit_least_squares has not been implemented.  Returning zeros.");
		return points;
	}
	
	public double solve_fixed_point(double x) {
		
        /*
         * 		def solve_fixed_point(f, x, ap=1e-6, rp=1e-4, ns=100):
         *			def g(x): return f(x)+x # f(x)=0 <=> g(x)=x
         *			Dg = D(g)
         *			for k in xrange(ns):
         *   			if abs(Dg(x)) >= 1:
         *       			raise ArithmeticError, 'error D(g)(x)>=1'
         *   			(x_old, x) = (x, g(x))
         *   		if k>2 and norm(x_old-x)<max(ap,norm(x)*rp):
         *       		return x
         *			raise ArithmeticError, 'no convergence'
         */
		// Variable declaration
		int k;				// Loop counting variable
		double dg;			// First derivative of G
		double x_old;		// Placeholder for previous value of x
		
		x_old = 0;
		dg = Dg(x);
		for(k=0; k<ns; k++) {
			if(Math.abs(dg)>=1) {
				System.out.println("Arithmatic error! Dg(x)>=1! Returning zero.");
				return 0;
			}
			x_old = x;
			x = g(x);
		}
		if (k>2 && A.norm(x_old-x)<Math.max(ap,A.norm(x)*rp)) {
			return x;
		}
		System.out.println("There is no convergence for solve_fixed_point. Returning zero.");
		return 0;
	}
	
	public double solve_bisection(double a, double b) {
		/*
		 * 		def solve_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
	     *			fa, fb = f(a), f(b)
	     *			if fa == 0: return a
	     *			if fb == 0: return b
	     *			if fa*fb > 0:
	     *			   raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
	     *			for k in xrange(ns):
	     *				x = (a+b)/2
	     *				fx = f(x)
	     *			    if fx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
	     *		 	  	elif fx * fa < 0: (b,fb) = (x, fx)
	     *   			else: (a,fa) = (x, fx)
	     *			raise ArithmeticError, 'no convergence'
	     */
		
		// Variable declaration
		double fa;		// The function with a
		double fb;		// The function with b
		double fx;		// The function with x
		int k;			// Loop counting variable
		double x;		// Variable to pass to the function
		
		fa=f(a);
		fb=f(b);
		if(fa==0) return a;
		if(fb==0) return b;
		if(fa*fb>0) {
			System.out.println("Arithmetic error! f(a) and f(b) must have opposite signs! Returning zero.");
			return 0;
		}
		for(k=0; k<ns; k++) {
			x = (a+b)/2;
			fx = f(x);
			if(fx==0 || A.norm(b=a)<Math.max(ap,A.norm(x)*rp)) return x;
			else if(fx*fa<0) {
				b=x;
				fb=fx;
			}
			else {
				a = x;
				fa = fx;
			}
		}
		System.out.println("Arithmetic error! No convergence for solve bisection! Returning zero.");
		return 0;
	}
    
	
	public double solve_newton(double x_guess) {
		/*
		 * 		def solve_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    	 *			x = float(x) # make sure it is not int
    	 *			for k in xrange(ns):
         *				(fx, Dfx) = (f(x), D(f)(x))
         *				if norm(Dfx) < ap:
         *   				raise ArithmeticError, 'unstable solution'
         *				(x_old, x) = (x, x-fx/Dfx)
         *				if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
    	 *			raise ArithmeticError, 'no convergence'
		 */
		
		// Variable declaration
		double x_old;	// Previous value of x
		double x;		// Current value of x
		
		x = x_guess;
    	for(int k=0; k<ns; k++) {
    		x_old = x;
      		x = x - f(x)/Df(x);
      		if(Math.abs(x-x_old)<Math.max(ap,rp*Math.abs(x))) return x;
    	}
    	System.out.println("Cannot solve Newton.Function does not converge. Returning zero.");
    	return 0;
	}  
	
    /*
     * 		def solve_secant(f, x, ap=1e-6, rp=1e-4, ns=20):
     *			x = float(x) # make sure it is not int
     *			(fx, Dfx) = (f(x), D(f)(x))
     *			for k in xrange(ns):
     *   			if norm(Dfx) < ap:
     *       			raise ArithmeticError, 'unstable solution'
     *   				(x_old, fx_old,x) = (x, fx, x-fx/Dfx)
     *   				if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
     *   				fx = f(x)
     *   				Dfx = (fx-fx_old)/(x-x_old)
     *			raise ArithmeticError, 'no convergence'
     */
	
    /*
     * 		def solve_newton_stabilized(f, a, b, ap=1e-6, rp=1e-4, ns=20):
     *			fa, fb = f(a), f(b)
     *			if fa == 0: return a
     *			if fb == 0: return b
     *			if fa*fb > 0:
     *   			raise ArithmeticError, 'f(a) and f(b) must have opposite sign'
     *			x = (a+b)/2
     *			(fx, Dfx) = (f(x), D(f)(x))
     *			for k in xrange(ns):
     *   			x_old, fx_old = x, fx
     *   			if norm(Dfx)>ap: x = x - fx/Dfx
     *   			if x==x_old or x<a or x>b: x = (a+b)/2
     *   			fx = f(x)
     *   			if fx==0 or norm(x-x_old)<max(ap,norm(x)*rp): return x
     *   			Dfx = (fx-fx_old)/(x-x_old)
     *   			if fx * fa < 0: (b,fb) = (x, fx)
     *   			else: (a,fa) = (x, fx)
     *			raise ArithmeticError, 'no convergence'
     */
	
    /*
     * 		def optimize_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     *			Dfa, Dfb = D(f)(a), D(f)(b)
     *			if Dfa == 0: return a
     *			if Dfb == 0: return b
     *			if Dfa*Dfb > 0:
     *   			raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
     *			for k in xrange(ns):
     *   			x = (a+b)/2
     *   			Dfx = D(f)(x)
     *   			if Dfx==0 or norm(b-a)<max(ap,norm(x)*rp): return x
     *   			elif Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
     *   			else: (a,Dfa) = (x, Dfx)
     *			raise ArithmeticError, 'no convergence'
     */
    
	double optimize_newton(double x_guess) {
		/*
		 * 		def optimize_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    	 *			x = float(x) # make sure it is not int
    	 *			for k in xrange(ns):
         *				(Dfx, DDfx) = (D(f)(x), DD(f)(x))
         *				if Dfx==0: return x
         *				if norm(DDfx) < ap:
         *   				raise ArithmeticError, 'unstable solution'
         *				(x_old, x) = (x, x-Dfx/DDfx)
         *				if norm(x-x_old)<max(ap,norm(x)*rp): return x
    	 *			raise ArithmeticError, 'no convergence'
		 */
		// Variable declaration
		double x_old;	// Previous value of x
		double x;		// Current value of x
		
		x = x_guess;
		for(int k=0; k<ns; k++) {
			x_old = x;
			x = x - Df(x)/DDf(x);
			if(Math.abs(x-x_old)<Math.max(ap,rp*Math.abs(x))) return x;
		}
		System.out.println("Cannot solve Newton.Function does not converge. Returning zero.");
		return 0;
	}  
	
	/*
	 * 		def optimize_secant(f, x, ap=1e-6, rp=1e-4, ns=100):
     *			x = float(x) # make sure it is not int
     *			(fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
     *			for k in xrange(ns):
     *		    if Dfx==0: return x
     *			if norm(DDfx) < ap:
     *				raise ArithmeticError, 'unstable solution'
     *   			(x_old, Dfx_old, x) = (x, Dfx, x-Dfx/DDfx)
     *   			if norm(x-x_old)<max(ap,norm(x)*rp): return x
     *   			fx = f(x)
     *   			Dfx = D(f)(x)
     *   			DDfx = (Dfx - Dfx_old)/(x-x_old)
     *			raise ArithmeticError, 'no convergence'
	 */
	
	/*
	 * 		def optimize_newton_stabilized(f, a, b, ap=1e-6, rp=1e-4, ns=20):
     *			Dfa, Dfb = D(f)(a), D(f)(b)
     *			if Dfa == 0: return a
     *			if Dfb == 0: return b
     *			if Dfa*Dfb > 0:
     *   			raise ArithmeticError, 'D(f)(a) and D(f)(b) must have opposite sign'
     *			x = (a+b)/2
     *			(fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
     *			for k in xrange(ns):
     *   			if Dfx==0: return x
     *   			x_old, fx_old, Dfx_old = x, fx, Dfx
     *   			if norm(DDfx)>ap: x = x - Dfx/DDfx
     *   			if x==x_old or x<a or x>b: x = (a+b)/2
     *   			if norm(x-x_old)<max(ap,norm(x)*rp): return x
     *   			fx = f(x)
     *   			Dfx = (fx-fx_old)/(x-x_old)
     *   			DDfx = (Dfx-Dfx_old)/(x-x_old)
     *   			if Dfx * Dfa < 0: (b,Dfb) = (x, Dfx)
     *   			else: (a,Dfa) = (x, Dfx)
     *			raise ArithmeticError, 'no convergence'
	 */
	
	/*
	 * 		def optimize_golden_search(f, a, b, ap=1e-6, rp=1e-4, ns=100):
     *			a,b=float(a),float(b)
     *			tau = (sqrt(5.0)-1.0)/2.0
     *			x1, x2 = a+(1.0-tau)*(b-a), a+tau*(b-a)
     *			fa, f1, f2, fb = f(a), f(x1), f(x2), f(b)
     *			for k in xrange(ns):
     *   			if f1 > f2:
     *       			a, fa, x1, f1 = x1, f1, x2, f2
     *       			x2 = a+tau*(b-a)
     *       			f2 = f(x2)
     *   			else:
     *       			b, fb, x2, f2 = x2, f2, x1, f1
     *       			x1 = a+(1.0-tau)*(b-a)
     *       			f1 = f(x1)
     *   			if k>2 and norm(b-a)<max(ap,norm(b)*rp): return b
     *			raise ArithmeticError, 'no convergence'
	 */
    
}
