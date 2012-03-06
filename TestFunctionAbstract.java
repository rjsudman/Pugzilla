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
	
	public abstract double f(double x);
	
	public double Df(double x) {
		return (f(x+h)-f(x-h))/(2.0*h);
	}

	public double DDf(double x) {
		return (f(x+h)-2.0*f(x)+f(x-h))/(h*h);
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

}
