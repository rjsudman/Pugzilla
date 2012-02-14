\\***************************************************************************\\
\\ Ruthann Sudman															 \\
\\ Final Project															 \\
\\ CSC 431-510																 \\
\\ March 17, 2012															 \\
\\ BSD License																 \\
\\ Original Python code composed by Massimo Di Pierro with BSD licensing	 \\
\\ # Created by Massimo Di Pierro - BSD License                              \\
\\***************************************************************************\\



public class MyMatrix {

	  public int rows;
	  public int cols;
	  public vector<float> data;
	  
	  public float abs(float x) {
		  if(x<0) return -x;
		  return x;
		}	  
	  
	  private void Matrix(int rows, int cols) {
	    this->rows=rows;
	    this->cols=cols;
	    this->data.resize(rows*cols);
	    for(int r=0; r<rows; r++)
	      for(int c=0; c<cols; c++)
		this->data[r*cols+c]=0;
	  }
	  
	  private float operator()(int i, int j) const {
	    return data[i*cols+j];
	  }
	  
	  private float &operator()(int i, int j) {
	    return data[i*cols+j];
	  }

	ostream &operator<<(ostream &out, const Matrix& A) {
		  out << "[";
		  for(int r=0; r<A.rows; r++) {
		    if(r>0) out << ",";
		    out << "[";
		    for(int c=0; c<A.cols; c++) {
		      if(c>0) out << ",";
		      out << A(r,c);    
		    }
		    out << "]";
		  }
		  out << "]";    
		  return out;
		}

		Matrix operator+(const Matrix &A, const Matrix &B) {
		  if(A.rows!=B.rows || A.cols!=B.cols)
		    cout << "BAD\n";
		  Matrix C(A.rows,A.cols);
		  for(int r=0; r<A.rows; r++)
		    for(int c=0; c<A.cols; c++)
		      C(r,c)=A(r,c)+B(r,c);
		  return C;
		}

		Matrix operator-(const Matrix &A, const Matrix &B) {
		  if(A.rows!=B.rows || A.cols!=B.cols)
		    cout << "BAD\n";
		  Matrix C(A.rows,A.cols);
		  for(int r=0; r<A.rows; r++)
		    for(int c=0; c<A.cols; c++)
		      C(r,c)=A(r,c)-B(r,c);
		  return C;
		}

		Matrix operator*(float a, const Matrix &B) {
		  Matrix C(B.rows,B.cols);
		  for(int r=0; r<B.rows; r++)
		    for(int c=0; c<B.cols; c++)
		      C(r,c)=a*B(r,c);
		  return C;
		}

		Matrix operator*(const Matrix &A, const Matrix &B) {
		  if(A.cols!=B.rows)
		    cout << "BAD\n";
		  Matrix C(A.rows,B.cols);
		  for(int r=0; r<A.rows; r++)
		    for(int c=0; c<B.cols; c++)
		      for(int k=0; k<A.cols; k++)
			C(r,c)+=A(r,k)*B(k,c);
		  return C;
		}

		void swap(float&a, float &b) {
		  float c=a; a=b; b=c;
		}

		Matrix inv(Matrix A) {
		  if(A.cols!=A.rows)
		    cout << "BAD\n";
		  Matrix B(A.rows,A.cols);
		  float p;
		  float q;
		  int m;
		  for(int r=0; r<B.cols;r++) B(r,r)=1;
		  for(int c=0; c<A.cols;c++) {    
		    m=c; p=A(c,c);
		    for(int i=c+1; i<A.rows; i++)
		      if(abs(A(i,c)) > abs(p)) {m=i; p=A(i,c);}
		    for(int i=0; i<A.cols; i++) {
		      swap(A(m,i),A(c,i));
		      swap(B(m,i),B(c,i));
		    }
		    for(int i=0; i<A.cols; i++) {
		      A(c,i) /= p; 
		      B(c,i) /= p;
		    }
		    for(int r=0; r<A.rows; r++) 
		      if(r!=c) {
			q = A(r,c);
			for(int i=0; i<A.cols; i++) {
			  A(r,i)-=q*A(c,i);
			  B(r,i)-=q*B(c,i);
			}
		      }
		  }
		  return B;
		}

		public static void main (String[] args) {
		  Matrix A(3,3);
		  Matrix B(3,3);
		  A(0,0)=1; A(0,1)=2; A(0,2)=3;
		  A(1,0)=1; A(1,1)=0; A(1,2)=3;
		  A(2,0)=2; A(2,1)=2; A(2,2)=4;
		  B = inv(A);
		  cout << B*A << endl;
		  return 0;
		}

}



