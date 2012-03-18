/*
 * TestFunctionAbstract.java
 * BSD License
 * Original Python code created by Massimo Di Pierro - BSD license	
 * Java implementation by Ruthann Sudman - BSD license
 * Repository at: https://github.com/rjsudman/Pugzilla
 */		

/** 
 * An abstract method requiring extension of the method f (a function) which
 * can then be used with this library of algorithms originally created in Python
 * by Massimo Di Pierro and ported to Java.   All code released under BSD 
 * licensing.
 * 
 * @author					Ruthann Sudman
 * @version					0.1
 * @see <a href="https://github.com/rjsudman/Pugzilla">Code Repository</a>
 */
public abstract class TestFunctionAbstract {
    
	public static double h=0.00001;		// Default h
	public static double ap=0.00001;	// Default absolute precision
	public static double rp=0.0001;		// Default relative precision
	public static int ns=100;			// Default number of steps
	private static LinearAlgebra A = new LinearAlgebra();
	
	/**
	 * An abstract function method to be extended by daughter classes.
	 * 
	 * @param x	The value used to evaluate the function.
	 * exceptions No known exceptions.
	 * @return The result of evaluating the function for x.
	 */
	public abstract double f(double x);
	
	/**
	 * The first derivative for the abstract function f.
	 * 
	 * @param x	The value used to evaluate the first derivative.
	 * @exception ArithmeticException No known exceptions.
	 * @return	The result of evaluating the first derivative for x.
	 */
	public double Df(double x) {
		
		try {
			return (f(x+h)-f(x-h))/(2.0*h);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **Df**!" + e.getMessage());
			return 0;
		}
	}
    
	/**
	 * The second derivative for the abstract function f.
	 * 
	 * @param x	The value used to evaluate the second derivative.
	 * @exception ArithmeticException No known exceptions.
	 * @return The result of evaluating the second derivative for x.
	 */
	public double DDf(double x) {
		
		try {
			return (f(x+h)-2.0*f(x)+f(x-h))/(h*h);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **DDf**!" + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * The abstract function f plus x
	 * 
	 * @param x	The value used to evaluate the second derivative.
	 * @exception ArithmeticException No known exceptions.
	 * @return	The result of evaluating the new function for x.
	 */
	public double g(double x) {
		
		try {
			return f(x)+x;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **g(x)**!" + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * The first derivative of the function g.
	 * 
	 * @param x The value used to evaluate the first derivative of g.
	 * @exception ArithmeticException No known exceptions.
	 * @return The result of evaluating the first derivative of g
	 */
	public double Dg(double x) {
		
		try {
			return (f(x+h)-f(x-h))/(2.0*h)+x;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **Dg**!" + e.getMessage());
			return 0;
		}
	}
    
	/**
	 * Evaluates the condition number of the abstract function f.
	 * 
	 * @param x	The value used to evaluate the condition number.
	 * @exception ArithmeticException Does not work when the f(x) evaluates to zero.
	 * @return The condition number for the abstract function f.
	 */
	public double condition_number(double x) {
		/* 
		 * 	def condition_number(f,x=None,h=1e-6):
		 * 		if callable(f) and not x is None:
		 *      	return D(f,h)(x)*x/f(x)
		 *  	elif isinstance(f,Matrix): # if is the Matrix Jƒzz
		 *      	return norm(f)*norm(1/f)
		 *  	else:
		 *      	raise NotImplementedError
		 */
		
		try {
			return Df(x)*x/f(x);
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **condition_number**!" + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * Evaluates the condition number of the TestMatrix f.
	 * 
	 * @param f	The TestMatrix to be evaluated for condition number.
	 * @exception ArithmeticException This function has not been properly implemented.
	 * @return The condition number for the TestMatrix f.
	 * @see TestMatrix
	 */
	public double condition_number(TestMatrix f) {
		
		// Variable declaration
		TestMatrix conditionMe;		// The condition matrix
		double conditionMe2;		// The condition number

		try {
			conditionMe = f.invMatrix();
			conditionMe2 = A.norm(f)*A.norm(conditionMe);
			return conditionMe2;
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **condition_number**!" + e.getMessage());
			return 0;
		}
	}
	
	/**
	 * Evaluates the abstract function f for fit least squares.
	 * 	
	 * @return A TestMatrix of the least squares fit
	 * @exception ArithmeticException This function has not been properly implemented, returns 0.
	 * @see TestMatrix
	 */
	public TestMatrix fit_least_squares() {
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
		 *		A = Matrix(len(points),len(f))
		 *		b = Matrix(len(points))
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
		
		System.err.println("Arithmetic Exception in **fit_least_squares**! Method not implemented. Returning zero.");
		return points;
	}
	
	/**
	 * Solves fixed point for the abstract function f.
	 * 
	 * @param x The value used to solve fixed point.
	 * @exception ArithmeticException Does not work when the first derivative is greater than or equal to 1. Does not work if fixed point does not converge for x.
	 * @return	Fixed point of the abstract function f for x.
	 */
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
	
		try {
			x_old = 0;
			dg = Dg(x);
			for(k=0; k<ns; k++) {
				if(Math.abs(dg)>=1) {
					System.err.println("Arithmatic Error! Dg(x)>=1 for **solve_fixed_point**. Returning zero.");
					return 0;
				}
				x_old = x;
				x = g(x);
			}
			if (k>2 && A.norm(x_old-x)<Math.max(ap,A.norm(x)*rp)) {
				return x;
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **solve_fixed_point**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **solve_fixed_point** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Solves bisection for the abstract function f.
	 * 
	 * @param a	The low value to examine the function.
	 * @param b	The high value to examine the function.
	 * @exception ArithmeticException f(a) and f(b) must have opposite signs. Does not work when bisection does not converge for f in range (a,b).
	 * @return Bisection for abstract function f in (a,b).
	 */
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
		
		try {
			fa=f(a);
			fb=f(b);
			if(fa==0) return a;
			if(fb==0) return b;
			if(fa*fb>0) {
				System.err.println("Arithmetic error! f(a) and f(b) must have opposite signs for **solve_bisection**. Returning zero.");
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
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **solve_bisection**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **solve_bisection** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Solves newton for the abstract function f.
	 * 
	 * @param x_guess	The result guess for newton.
	 * @exception ArithmeticException Does not work when newton does not converge for f in x.
	 * @return Newton for abstract function f in x.
	 */
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
		
		try {
			x = x_guess;
	    	for(int k=0; k<ns; k++) {
	    		x_old = x;
	      		x = x - f(x)/Df(x);
	      		if(Math.abs(x-x_old)<Math.max(ap,rp*Math.abs(x))) return x;
	    	}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **solve_newton**!" + e.getMessage());
			return 0;
		}
    	System.err.println("Arithmetic Error! **solve_newton** does not converge. Returning zero.");
    	return 0;

	}  
	
	/**
	 * Solves secant for the abstract function f.
	 * 
	 * @param x	The value used to evaluate the abstract function f for secant.
	 * @exception ArithmeticException If the norm of the function is less than the absolute function. If the secant does not converge for abstract function f in x.
	 * @return	Secant of the abstract function f in x.
	 */
	public double solve_secant(double x) {
		/*
		 * 		def solve_secant(f, x, ap=1e-6, rp=1e-4, ns=20):
	     *			x = float(x) # make sure it is not int
	     *			(fx, Dfx) = (f(x), D(f)(x))
	     *			for k in xrange(ns):
	     *   			if norm(Dfx) < ap:
	     *       			raise ArithmeticError, 'unstable solution'
	     *   			(x_old, fx_old,x) = (x, fx, x-fx/Dfx)
	     *   			if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
	     *   			fx = f(x)
	     *   			Dfx = (fx-fx_old)/(x-x_old)
	     *			raise ArithmeticError, 'no convergence'
	     */
		
		// Variable declaration
		int k;			// Loop counting variable
		double fx;		// The result of the function
		double Dfx;		// The result of the derivative of the function
		double x_old;	// Previous value of x
		double fx_old;	// Previous value of f(x)
		
		try {
			fx = f(x);
			Dfx = Df(x);
			for(k=0;k<ns; k++) {
				if(A.norm(Dfx) < ap) {
					System.err.println("Arithmetic Error! Unstable solution for **solve_secant**.  Returning zero.");
					return 0;
				}
				x_old = x;
				fx_old = fx;
				x = x-fx/Dfx;
				if(k>2 && A.norm(x-x_old)<Math.max(ap, A.norm(x)*rp)) return x;
				fx = f(x);
				Dfx = (fx-fx_old)/(x-x_old);
			}
		}	
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **solve_secant**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **solve_secant** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Solves newton stabilized for the abstract function f in (a,b).
	 * 
	 * @param a	The low value for f.
	 * @param b	The high value for f.
	 * @exception ArithmeticException f(a) and f(b) must evaluate with opposite signs. Does not work if newton stabilized does not converge.
	 * @return Newton stabilized for the abstract function f in (a,b).
	 */
	public double solve_newton_stabilized(double a, double b) {
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
        
		// Variable declaration
		int k;			// Loop counting variable
		double x;		// Average of a and b
		double fa;		// The result of f(a)
		double fb;		// The result of f(b)
		double fx;		// The result of the function
		double Dfx;		// The result of the derivative of the function
		double x_old;	// Previous value of x
		double fx_old;	// Previous value of f(x)
        
		try {
			fa = f(a);
			fb = f(b);
			if(fa ==0) return a;
			if(fb==0) return b;
			if (fa*fb > 0) {
				System.err.println("Arithmetic Error! f(a) and f(b) must have opposite sign for **solve_newton_stabilized**. Returning zero.");
				return 0;
			}
			x=(a+b)/2;
			fx = f(x);
			Dfx = Df(x);
			for(k=0; k<ns; k++) {
				x_old = x;
				fx_old = fx;
				if (A.norm(Dfx)>ap) {
					x = x-fx/Dfx;
				}
				if(x==x_old || x<a || x>b) {
					x=(a+b)/2;
				}
				fx = f(x);
				if(fx==0 || A.norm(x-x_old)<Math.max(ap, A.norm(x)*rp)) return x;
				Dfx = (fx-fx_old)/(x-x_old);
				if(fx*fa<0) {
					b = x;
					fb = fx;
				}
				else {
					a = x;
					fa = fx;
				}
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **solve_newton_stabilized**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **solve_newton_stabilized** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Optimized bisection for the abstract function f in (a,b).
	 * 
	 * @param a	The low value.
	 * @param b	The high value.
	 * @exception ArithmeticException Df(a) and Df(b) must evaluate with opposite signs. Does not work when bisection does not converge for f in (a,b).
	 * @return Optimized bisection for the abstract function f in (a,b).
	 */
	public double optimize_bisection(double a, double b) {
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
        
		// Variable declaration
		int k;			// Loop counting variable
		double x;		// Average of a and b
		double Dfa;		// The result of Df(a)
		double Dfb;		// The result of Df(b)
		double Dfx;		// The result of the derivative of the function
        
		try {
			Dfa = Df(a);
			Dfb = Df(b);
			if(Dfa==0) return a;
			if(Dfb==0) return b;
			if(Dfa*Dfb > 0) {
				System.err.println("Arithmetic Error! Df(a) and Df(b) must have opposite sign for **optimize_bisection**. Return zero.");
				return 0;
			}
			for(k=0; k<ns; k++) {
				x = (a+b)/2;
				Dfx = Df(x);
				if(Dfx==0 || A.norm(b-a)<Math.max(ap,A.norm(x)*rp)) return x;
				else if(Dfx*Dfa<0) {
					b = x;
					Dfb = Dfx;
				}
				else {
					a = x;
					Dfa = Dfx;
				}
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **optimize_bisection**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **optimize_bisection** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Newton optimized for the abstract function f.
	 * 
	 * @param x_guess The guess for newton.
	 * @exception ArithmeticException Does not work if newton does not converge for f in x.
	 * @return Newton optimized for the abstract function f in x.
	 */
	public double optimize_newton(double x_guess) {
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
		
		try {
			x = x_guess;
			for(int k=0; k<ns; k++) {
				x_old = x;
				x = x - Df(x)/DDf(x);
				if(Math.abs(x-x_old)<Math.max(ap,rp*Math.abs(x))) return x;
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **optimize_newton**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **optimize_newton** does not converge. Returning zero.");
		return 0;

	}  
	
	/**
	 * Optimized secant for the abstract function f.
	 * 
	 * @param x The value used to evaluate secant for f.
	 * @exception ArithmeticException Does not work if DDf(x) is less than absolute precision. Does not work if optimize secant does not converge for f in x.
	 * @return Optimized secant for the abstract function f.
	 */
	public double optimize_secant(double x) {
		/*
		 * 		def optimize_secant(f, x, ap=1e-6, rp=1e-4, ns=100):
	     *			x = float(x) # make sure it is not int
	     *			(fx, Dfx, DDfx) = (f(x), D(f)(x), DD(f)(x))
	     *			for k in xrange(ns):
	     *		    	if Dfx==0: return x
	     *				if norm(DDfx) < ap:
	     *					raise ArithmeticError, 'unstable solution'
	     *   			(x_old, Dfx_old, x) = (x, Dfx, x-Dfx/DDfx)
	     *   			if norm(x-x_old)<max(ap,norm(x)*rp): return x
	     *   			fx = f(x)
	     *   			Dfx = D(f)(x)
	     *   			DDfx = (Dfx - Dfx_old)/(x-x_old)
	     *			raise ArithmeticError, 'no convergence'
		 */
		
		// Variable declaration
		int k;			// Loop counting variable
		double Dfx;		// The result of the derivative of the function
		double DDfx;	// The result of the second derivative of the function
		double x_old;	// Previous value of x
		double Dfx_old;	// Previous value of Df(x)
        
		try {
			Dfx = Df(x);
			DDfx = DDf(x);
			for(k=0; k<ns; k++) {
				if(Dfx==0) return x;
				if(A.norm(DDfx) <ap) {
					System.out.println("Arithmetic Error. Unstable solution for **optimize_secant**. Returning zero.");
					return 0;
				}
				x_old = x;
				Dfx_old = Dfx;
				x = x-Dfx/DDfx;
				if((x-x_old)<Math.max(ap, A.norm(x)*rp)) return x;
				Dfx = Df(x);
				DDfx = (Dfx - Dfx_old)/(x-x_old);
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **optimize_secant**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **optimize_secant** does not converge. Returning zero.");
		return 0;
	}
	
	/**
	 * Optimization of newton stabilized for the abstract function f.
	 * 
	 * @param a	The low value.
	 * @param b	The high value.
	 * @exception ArithmeticException Df(a) and Df(b) must evaluate with opposite signs. Does not work if newton does not converge for the abstract function f in (a,b).
	 * @return Optimized newton stabilized for the abstract function f.
	 */
	public double optimize_newton_stabilized(double a, double b) {
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
        
		// Variable declaration
		int k;				// Loop counting variable
		double x;			// Average of a and b
		double Dfa;			// The result of Df(a)
		double Dfb;			// The result of Df(b)
		double fx;			// The result of the function
		double Dfx;			// The result of the derivative of the function
		double DDfx;		// The result of the second derivative of the function
		double x_old;		// Previous value of x
		double fx_old;		// Previous value of f(x)
		double Dfx_old;		// Previous value of Df(x)
        
		try {
			Dfa = Df(a);
			Dfb = Df(b);
			if (Dfa == 0) return a;
			if(Dfb == 0) return b;
			if (Dfa*Dfb>0) {
				System.err.println("Arithmetic Error! Df(a) and Df(b) must have opposite sign for **optimize_newton_stabilized**. Returning zero.");
				return 0;
			}
			x=(a+b)/2;
			fx = f(x);
			Dfx = Df(x);
			DDfx = DDf(x);
			for(k=0; k<ns; k++) {
				if(Dfx ==0) return x;
				x_old = x;
				fx_old = fx;
				Dfx_old = Dfx;
				if (A.norm(DDfx)>ap) {
					x = x-Dfx/DDfx;
				}
				if(x==x_old || x<a || x>b) {
					x=(a+b)/2;
				}
				if(A.norm(x-x_old)<Math.max(ap, A.norm(x)*rp)) return x;
				fx = f(x);
				Dfx = (fx-fx_old)/(x-x_old);
				DDfx = (Dfx - Dfx_old)/(x-x_old);
				if(Dfx*Dfa<0) {
					b = x;
					Dfb = Dfx;
				}
				else {
					a = x;
					Dfa = Dfx;
				}
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in **optimize_newton_stabilized**!" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **optimize_newton_stabilized** does not converge. Returning zero.");
		return 0;
 	}
	
	/**
	 * Optimizes golden search for the abstract function f in (a,b).
	 * 
	 * @param a The low value.
	 * @param b	The high value.
	 * @exception ArithmeticException Does not work if golden search cannot be optimized for the abstract function f in (a,b).
	 * @return The optimized golden search for abstract function f.
	 */
	public double optimize_golden_search(double a, double b) {
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
        
		// Variable declaration
		int k;			// Loop counting variable
		double tau;		// tau
		double x1;		// Guess 1
		double x2;		// Guess 2
		double f1;		// f(x1)
		double f2;		// f(x2)
        
		try {
			tau = (Math.sqrt(5.0)-1)/2;
			x1 = a+(1-tau)*(b-a);
			x2 = a+tau*(b-a);
			f1 = f(x1);
			f2 = f(x2);
			for(k=0; k<ns; k++) {
				if(f1>f2) {
					a = x1;
					x1 = x2;
					f1 = f2;
					x2 = a+tau*(b-a);
					f2 = f(x2);
				}
				else {
					b = x2;
					x2 = x1;
					f2 = f1;
					x1 = a+(1.0-tau)*(b-a);
					f1 = f(x1);
				}
				if(k>2 && A.norm(b-a)<Math.max(ap, A.norm(b)*rp)) return b;
			}
		}
		catch (ArithmeticException e) {
			System.err.println("Arithmetic exception in !" + e.getMessage());
			return 0;
		}
		System.err.println("Arithmetic Error! **optimize_golden_search** does not converge. Returning zero.");
		return 0;
	}
}
