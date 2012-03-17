# Created by Massimo Di Pierro - BSD License

class YStock:
    """
    Class that downloads and stores data from Yahoo Finance
    Examples:
    >>> google = YStock('GOOG')
    >>> current = google.current()
    >>> price = current['price']
    >>> market_cap = current['market_cap']
    >>> h = google.historical()
    >>> last_adjusted_close = h[-1]['adjusted_close']
    >>> last_log_return = h[-1]['log_return']
    """
    URL_CURRENT = 'http://finance.yahoo.com/d/quotes.csv?s=%(symbol)s&f=%(columns)s'
    URL_HISTORICAL = 'http://ichart.yahoo.com/table.csv?s=%(s)s&a=%(a)s&b=%(b)s&c=%(c)s&d=%(d)s&e=%(e)s&f=%(f)s'
    def __init__(self,symbol):
        self.symbol = symbol.upper()

    def current(self):
        import urllib
        FIELDS = (('price', 'l1'),
                  ('change', 'c1'),
                  ('volume', 'v'),
                  ('average_daily_volume', 'a2'),
                  ('stock_exchange', 'x'),
                  ('market_cap', 'j1'),
                  ('book_value', 'b4'),
                  ('ebitda', 'j4'),
                  ('dividend_per_share', 'd'),
                  ('dividend_yield', 'y'),
                  ('earnings_per_share', 'e'),
                  ('52_week_high', 'k'),
                  ('52_week_low', 'j'),
                  ('50_days_moving_average', 'm3'),
                  ('200_days_moving_average', 'm4'),
                  ('price_earnings_ratio', 'r'),
                  ('price_earnings_growth_ratio', 'r5'),
                  ('price_sales_ratio', 'p5'),
                  ('price_book_ratio', 'p6'),
                  ('short_ratio', 's7'))
        columns = ''.join([row[1] for row in FIELDS])
        url = self.URL_CURRENT % dict(symbol=self.symbol, columns=columns)
        raw_data = urllib.urlopen(url).read().strip().strip('"').split(',')
        current = dict()
        for i,row in enumerate(FIELDS):
            try:
                current[row[0]] = float(raw_data[i])
            except:
                current[row[0]] = raw_data[i]
        return current

    def historical(self,start=None, stop=None):
        import datetime, time, urllib, math
        start =  start or datetime.date(1900,1,1)
        stop = stop or datetime.date.today()
        url = self.URL_HISTORICAL % dict(
            s=self.symbol,
            a=start.month-1,b=start.day,c=start.year,
            d=stop.month-1,e=stop.day,f=stop.year)
        # Date,Open,High,Low,Close,Volume,Adj Close
        lines = urllib.urlopen(url).readlines()
        raw_data = [row.split(',') for row in lines[1:]]
        previous_adjusted_close = 0
        series = []
        raw_data.reverse()
        for row in raw_data:
            adjusted_close = float(row[6])
            if previous_adjusted_close:
                arithmetic_return = adjusted_close/previous_adjusted_close-1.0

                log_return = math.log(adjusted_close/previous_adjusted_close)
            else:
                arithmetic_return = log_return = None
            previous_adjusted_close = adjusted_close
            series.append(dict(
               date = datetime.datetime.strptime(row[0],'%Y-%m-%d'),
               open = float(row[1]),
               high = float(row[2]),
               low = float(row[3]),
               close = float(row[4]),
               volume = float(row[5]),
               adjusted_close = adjusted_close,
               arithmetic_return = arithmetic_return,
               log_return = log_return))
        return series

    @staticmethod
    def download(symbol='goog',what='adjusted_close',start=None,stop=None):
        return [d[what] for d in YStock(symbol).historical(start,stop)]

import os
import uuid
import sqlite3
import cPickle as pickle

class PersistentDictionary(object):
    """
    A sqlite based key,value storage.
    The value can be any pickable object.
    Similar interface to Python dict
    Supports the GLOB syntax in methods keys(),items(), __delitem__()

    Usage Example:
    >>> p = PersistentDictionary(path='test.sqlite')
    >>> key = 'test/' + p.uuid()
    >>> p[key] = {'a': 1, 'b': 2}
    >>> print p[key]
    {'a': 1, 'b': 2}
    >>> print len(p.keys('test/*'))
    1
    >>> del p[key]
    """

    CREATE_TABLE = "CREATE TABLE persistence (pkey, pvalue)"
    SELECT_KEYS = "SELECT pkey FROM persistence WHERE pkey GLOB ?"
    SELECT_VALUE = "SELECT pvalue FROM persistence WHERE pkey GLOB ?"
    INSERT_KEY_VALUE = "INSERT INTO persistence(pkey, pvalue) VALUES (?,?)"
    DELETE_KEY_VALUE = "DELETE FROM persistence WHERE pkey LIKE ?"
    SELECT_KEY_VALUE = "SELECT pkey,pvalue FROM persistence WHERE pkey GLOB ?"

    def __init__(self,
                 path='persistence.sqlite',
                 autocommit=True):
        self.path = path
        self.autocommit = autocommit
        create_table = not os.path.exists(path)
        self.connection  = sqlite3.connect(path)
        self.connection.text_factory = str # do not use unicode
        self.cursor = self.connection.cursor()
        if create_table:
            self.cursor.execute(self.CREATE_TABLE)
            self.connection.commit()

    def uuid(self):
        return str(uuid.uuid4())

    def keys(self,pattern='*'):
        "returns a list of keys filtered by a pattern, * is the wildcard"
        self.cursor.execute(self.SELECT_KEYS,(pattern,))
        return [row[0] for row in self.cursor.fetchall()]

    def __contains__(self,key):
        return True if self[key] else False

    def __iter__(self):
        for key in self:
            yield key

    def __setitem__(self,key,value):
        if value is None:
            del self[key]
            return
        self.cursor.execute(self.INSERT_KEY_VALUE,
                            (key, pickle.dumps(value)))
        if self.autocommit: self.connection.commit()

    def __getitem__(self,key):
        self.cursor.execute(self.SELECT_VALUE, (key,))
        row = self.cursor.fetchone()
        return pickle.loads(row[0]) if row else None

    def __delitem__(self,pattern):
        self.cursor.execute(self.DELETE_KEY_VALUE, (pattern,))
        if self.autocommit: self.connection.commit()

    def items(self,pattern='*'):
        self.cursor.execute(self.SELECT_KEY_VALUE, (pattern,))
        return [(row[0],pickle.loads(row[1])) \
                    for row in self.cursor.fetchall()]

import math
import cmath
import random
import os
import tempfile

os.environ['MPLCONfigureDIR'] = tempfile.mkdtemp()
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse
except ImportError:
    print 'warning: matplotlib not available'

def draw(title='title',xlab='x',ylab='y',filename='tmp.png',
         linesets=None, pointsets=None, histsets=None, ellisets=None,
         xrange=None, yrange=None):
    figure = Figure(frameon=False)
    figure.set_facecolor('white')
    axes = figure.add_subplot(111)
    axes.grid(True)
    if title: axes.set_title(title)
    if xlab: axes.set_xlabel(xlab)
    if ylab: axes.set_ylabel(ylab)
    if xrange: axes.set_xlim(xrange)
    if yrange: axes.set_ylim(yrange)
    legend = [],[]

    for histset in histsets or []:
        data = histset['data']
        bins = histset.get('bins',20)
        color = histset.get('color','blue')
        q = axes.hist(data,bins, color=color)
        if 'legend' in histset:
            legend[0].append(q)
            legend[1].append(histset['legend'])

    for lineset in linesets or []:
        data = lineset['data']
        color = lineset.get('color','black')
        linestyle = lineset.get('style','-')
        linewidth = lineset.get('width',2)
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        q = axes.plot(x, y, linestyle=linestyle,
                      linewidth=linewidth, color=color)
        if 'legend' in lineset:
            legend[0].append(q)
            legend[1].append(lineset['legend'])

    for pointset in pointsets or []:
        data = pointset['data']
        color = pointset.get('color','black')
        marker = pointset.get('marker','o')
        linewidth = pointset.get('width',2)
        x = [p[0] for p in data]
        y = [p[1] for p in data]
        yerr = [p[2] for p in data]
        q = axes.errorbar(x, y, yerr=yerr, fmt=marker,
                          linewidth=linewidth, color=color)
        if 'legend' in pointset:
            legend[0].append(q)
            legend[1].append(pointset['legend'])


    for elliset in ellisets or []:
        data = elliset['data']
        color = elliset.get('color','blue')
        for point in data:
            x, y = point[:2]
            dx = point[2] if len(point)>2 else 0.01
            dy = point[3] if len(point)>3 else dx
            ellipse = Ellipse(xy=(x,y),width=dx,height=dy)
            axes.add_artist(ellipse)
            ellipse.set_clip_box(axes.bbox)
            ellipse.set_alpha(0.5)
            ellipse.set_facecolor(color)

    if legend[0]: axes.legend(*legend)
    canvas = FigureCanvas(figure)
    canvas.print_png(open(filename,'wb'))

def color2d(title='title',xlab='x',ylab='y',
            data=[[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]],
            filename = 'tmp.png'):
    figure=Figure()
    figure.set_facecolor('white')
    axes=figure.add_subplot(111)
    if title: axes.set_title(title)
    if xlab: axes.set_xlabel(xlab)
    if ylab: axes.set_ylabel(ylab)
    image=axes.imshow(data)
    image.set_interpolation('bilinear')
    canvas = FigureCanvas(figure)
    canvas.print_png(open(filename,'wb'))

class memoize(object):
    def __init__ (self, f):
        self.f = f
        self.storage = {}
    def __call__ (self, *args, **kwargs):
        key = str((self.f.__name__, args, kwargs))
        try:
            value = self.storage[key]
        except KeyError:
            value = self.f(*args, **kwargs)
            self.storage[key] = value
        return value

@memoize
def fib(n):
    return 1 if n in (0,1) else fib(n-1)+fib(n-2)

class memoize_persistent(object):
    STORAGE = 'memoize.sqlite'
    def __init__ (self, f):
        self.f = f
        self.storage = PersistentDictionary(memoize_persistent.STORAGE)
    def __call__ (self, *args, **kwargs):
        key = str((self.f.__name__, args, kwargs))
        try:
            value = self.storage[key]
        except KeyError:
            value = self.f(*args, **kwargs)
            self.storage[key] = value
        return value

def breadth_first_search(graph,start):
    vertices, link = graph
    blacknodes = []
    graynodes = [start]
    neighbors = [[] for vertex in vertices]
    for link in links:
        neighbors[link[0]].append(link[1])
    while graynodes:
        current = graynodes.pop()
        for neighbor in neighbors[current]:
            if not neighbor in blacknodes+graynodes:
                graynodes.insert(0,neighbor)
        blacknodes.append(current)
    return blacknodes

def depth_first_search(graph,start):
    vertices, link = graph
    blacknodes = []
    graynodes = [start]
    neighbors = [[] for vertex in vertices]
    for link in links:
        neighbors[link[0]].append(link[1])
    while graynodes:
        current = graynodes.pop()
        for neighbor in neighbors[current]:
            if not neighbor in blacknodes+graynodes:
                graynodes.append(neighbor)
        blacknodes.append(current)
    return blacknodes

class DisjointSets(object):
    def __init__(self,n):
        self.sets = [-1]*n
        self.counter = n
    def parent(self,i):
        while True:
            j = self.sets[i]
            if j<0:
                return i
            i = j
    def join(self,i,j):
        i,j = self.parent(i),self.parent(j)
        if i!=j:
            self.sets[i] += self.sets[j]
            self.sets[j] = i
            self.counter-=1
            return True # they have been joined
        return False    # they were already joined
    def __len__(self):
        return self.counter

def make_maze(n,d):
    walls = [(i,i+n**j) for i in xrange(n**2) for j in xrange(d) if (i/n**j)%n+1<n]
    teared_down_walls = []
    ds = DisjointSets(n**d)
    random.shuffle(walls)
    for i,wall in enumerate(walls):
        if ds.join(wall[0],wall[1]):
            teared_down_walls.append(wall)
        if len(ds)==1:
            break
    walls = [wall for wall in walls if not wall in teared_down_walls]
    return walls, teared_down_walls

def Kruskal(graph):
    vertices, links = graph
    A = []
    S = DisjointSets(len(vertices))
    links.sort(cmp=lambda a,b: cmp(a[2],b[2]))
    for source,dest,length in links:
        if S.join(source,dest):
            A.append((source,dest,length))
    return A

class PrimVertex(object):
    INFINITY = 1e100
    def __init__(self,id,links):
        self.id = id
        self.closest = None
        self.closest_dist = PrimVertex.INFINITY
        self.neighbors = [link[1:] for link in links if link[0]==id]
    def __cmp__(self,other):
        return cmp(self.closest_dist, other.closest_dist)

def Prim(graph, start):
    from heapq import heappush, heappop, heapify
    vertices, links = graph
    P = [PrimVertex(i,links) for i in vertices]
    Q = [P[i] for i in vertices if not i==start]
    vertex = P[start]
    while Q:
        for neighbor_id,length in vertex.neighbors:
            neighbor = P[neighbor_id]
            if neighbor in Q and length<neighbor.closest_dist:
                 neighbor.closest = vertex
                 neighbor.closest_dist = length
        heapify(Q)
        vertex = heappop(Q)
    return [(v.id,v.closest.id,v.closest_dist) for v in P if not v.id==start]

def Dijkstra(graph, start):
    from heapq import heappush, heappop, heapify
    vertices, links = graph
    P = [PrimVertex(i,links) for i in vertices]
    Q = [P[i] for i in vertices if not i==start]
    vertex = P[start]
    vertex.closest_dist = 0
    while Q:
        for neighbor_id,length in vertex.neighbors:
            neighbor = P[neighbor_id]
            dist = length+vertex.closest_dist
            if neighbor in Q and dist<neighbor.closest_dist:
                 neighbor.closest = vertex
                 neighbor.closest_dist = dist
        heapify(Q)
        vertex = heappop(Q)
    return [(v.id,v.closest.id,v.closest_dist) for v in P if not v.id==start]

def encode_huffman(input):
    from heapq import heappush, heappop

    def inorder_tree_walk(t, key, keys):
        (f,ab) = t
        if isinstance(ab,tuple):
            inorder_tree_walk(ab[0],key+'0',keys)
            inorder_tree_walk(ab[1],key+'1',keys)
        else:
            keys[ab] = key

    symbols = {}
    for symbol in input:
        symbols[symbol] = symbols.get(symbol,0)+1
    heap = []
    for (k,f) in symbols.items():
        heappush(heap,(f,k))
    while len(heap)>1:
        (f1,k1) = heappop(heap)
        (f2,k2) = heappop(heap)
        heappush(heap,(f1+f2,((f1,k1),(f2,k2))))
    symbol_map = {}
    inorder_tree_walk(heap[0],'',symbol_map)
    encoded = ''.join(symbol_map[symbol] for symbol in input)
    return symbol_map, encoded

def decode_huffman(keys, encoded):
    reversed_map = dict((v,k) for (k,v) in keys.items())
    i, output = 0, []
    for j in range(1,len(encoded)+1):
        if encoded[i:j] in reversed_map:
           output.append(reversed_map[encoded[i:j]])
           i=j
    return ''.join(output)

def lcs(a, b):
    previous = [0]*len(a)
    for i,r in enumerate(a):
        current = []
        for j,c in enumerate(b):
            if r==c:
                e = previous[j-1]+1 if i*j>0 else 1
            else:
                e = max(previous[j] if i>0 else 0,
                        current[-1] if j>0 else 0)
            current.append(e)
        previous=current
    return current[-1]

def needleman_wunsch(a,b,p=0.97):
    z=[]
    for i,r in enumerate(a):
        z.append([])
        for j,c in enumerate(b):
            if r==c:
                e = z[i-1][j-1]+1 if i*j>0 else 1
            else:
                e = p*max(z[i-1][j] if i>0 else 0,
                          z[i][j-1] if j>0 else 0)
            z[-1].append(e)
    return z

def continuum_knapsack(a,b,c):
    table = [(a[i]/b[i],i) for i in range(len(a))]
    table.sort()
    table.reverse()
    f=0.0
    for (y,i) in table:
        quantity = min(c/b[i],1)
        x.append((i,quantity))
        c = c-b[i]*quantity
        f = f+a[i]*quantity
    return (f,x)

def D(f,h=1e-6): # first derivative of f
    return lambda x,f=f,h=h: (f(x+h)-f(x-h))/2/h

def DD(f,h=1e-6): # second derivative of f
    return lambda x,f=f,h=h: (f(x+h)-2.0*f(x)+f(x-h))/(h*h)

def myexp(x,precision=1e-6,max_steps=40):
    if x==0:
       return 1.0
    elif x>0:
       return 1.0/myexp(-x,precision,max_steps)
    else:
       t = s = 1.0 # first term
       for k in range(1,max_steps):
           t = t*x/k   # next term
           s = s + t   # add next term
           if abs(t)<precision: return s
       raise ArithmeticError, 'no convergence'

def mysin(x,precision=1e-6,max_steps=40):
    pi = math.pi
    if x==0:
       return 0
    elif x<0:
       return -mysin(-x)
    elif x>2.0*pi:
       return mysin(x % (2.0*pi))
    elif x>pi:
       return -mysin(2.0*pi - x)
    elif x>pi/2:
       return mysin(pi-x)
    elif x>pi/4:
       return sqrt(1.0-mysin(pi/2-x)**2)
    else:
       t = s = x                     # first term
       for k in range(1,max_steps):
           t = t*(-1.0)*x*x/(2*k)/(2*k+1)   # next term
           s = s + t                 # add next term
           r = x**(2*k+1)            # estimate residue
           if r<precision: return s  # stopping condition
       raise ArithmeticError, 'no convergence'

def mycos(x,precision=1e-6,max_steps=40):
    pi = math.pi
    if x==0:
       return 1.0
    elif x<0:
       return mycos(-x)
    elif x>2.0*pi:
       return mycos(x % (2.0*pi))
    elif x>pi:
       return mycos(2.0*pi - x)
    elif x>pi/2:
       return -mycos(pi-x)
    elif x>pi/4:
       return sqrt(1.0-mycos(pi/2-x)**2)
    else:
       t = s = 1                     # first term
       for k in range(1,max_steps):
           t = t*(-1.0)*x*x/(2*k)/(2*k-1)   # next term
           s = s + t                 # add next term
           r = x**(2*k)              # estimate residue
           if r<precision: return s  # stopping condition
       raise ArithmeticError, 'no convergence'

class Matrix(object):
    def __init__(self,rows=1,cols=1,fill=0.0,optimize=False):
        """
        Constructor a zero matrix
        Parameters
        - rows: the integer number of rows
        - cols: the integer number of columns
        - fill: the value or callable to be used to fill the matrix
        """
        self.rows = rows
        self.cols = cols
        if callable(fill):
            self.data = [fill(r,c) for r in xrange(rows) for c in xrange(cols)]
        else:
            self.data = [fill for r in xrange(rows) for c in xrange(cols)]
        if optimize:
            import array
            self.data = array.array('d',self.data)

    def __getitem__(self,(i,j)):
        return self.data[i*self.cols+j]

    def __setitem__(self,(i,j),value):
        self.data[i*self.cols+j] = value

    def row(A,i):
        return Matrix(self.cols,1,fill=lambda r,c: A[i,c])

    def col(self,i):
        return Matrix(self.rows,1,fill=lambda r,c: A[r,i])

    def as_list(A):
        return [[A[r,c] for c in xrange(A.cols)] for r in xrange(A.rows)]

    def __str__(self):
        return str(self.as_list())

    @staticmethod
    def identity(rows=1,one=1.0,fill=0.0):
        """
        Constuctor a diagonal matrix
        Parameters
        - rows: the integer number of rows (also number of columns)
        - fill: the value to be used to fill the matrix
        - one: the value in the diagonal
        """
        M = Matrix(rows,rows,fill)
        for i in xrange(rows): M[i,i] = one
        return M

    @staticmethod
    def diagonal(d):
        M = Matrix(len(d),len(d))
        for i,e in enumerate(d): M[i,i] = e
        return M

    @staticmethod
    def from_list(v):
        "builds a matrix from a list of lists"
        return Matrix(len(v),len(v[0]),fill=lambda r,c: v[r][c])

    def __add__(A,B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix.from_list([[4,3.0], [2,1.0]])
        >>> B = Matrix.from_list([[1,2.0], [3,4.0]])
        >>> C = A + B
        >>> print C
        [[5, 5.0], [5, 5.0]]
        """
        n, m = A.rows, A.cols
        if not isinstance(B,Matrix):
            if n==m:
                B = Matrix.identity(n,B)
            elif n==1 or m==1:
                B = Matrix(n,m,fill=B)
        if B.rows!=n or B.cols!=m:
            raise ArithmeticError, "Incompatible dimensions"
        C = Matrix(n,m)
        for r in xrange(n):
            for c in xrange(m):
                C[r,c] = A[r,c]+B[r,c]
        return C

    def __sub__(A,B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix.from_list([[4.0,3.0], [2.0,1.0]])
        >>> B = Matrix.from_list([[1.0,2.0], [3.0,4.0]])
        >>> C = A - B
        >>> print C
        [[3.0, 1.0], [-1.0, -3.0]]
        """
        n, m = A.rows, A.cols
        if not isinstance(B,Matrix):
            if n==m:
                B = Matrix.identity(n,B)
            elif n==1 or m==1:
                B = Matrix(n,m,fill=B)
        if B.rows!=n or B.cols!=m:
            raise ArithmeticError, "Incompatible dimensions"
        C = Matrix(n,m)
        for r in xrange(n):
            for c in xrange(m):
                C[r,c] = A[r,c]-B[r,c]
        return C
    def __radd__(A,B): #B+A
        return A+B
    def __rsub__(A,B): #B-A
        return (-A)+B
    def __neg__(A):
        return Matrix(A.rows,A.cols,fill=lambda r,c:-A[r,c])

    def __rmul__(A,x):
        "multiplies a number of matrix A by a scalar number x"
        import copy
        M = copy.deepcopy(A)
        for r in xrange(M.rows):
            for c in xrange(M.cols):
                 M[r,c] *= x
        return M

    def __mul__(A,B):
        "multiplies a number of matrix A by another matrix B"
        if isinstance(B,(list,tuple)):
            return (A*Matrix(len(B),1,fill=lambda r,c:B[r])).data
        elif not isinstance(B,Matrix):
            return B*A
        elif A.cols == 1 and B.cols==1 and A.rows == B.rows:
            # try a scalar product ;-)
            return sum(A[r,0]*B[r,0] for r in xrange(A.rows))
        elif A.cols!=B.rows:
            raise ArithmeticError, "incompatible dimensions"
        M = Matrix(A.rows,B.cols)
        for r in xrange(A.rows):
            for c in xrange(B.cols):
                for k in xrange(A.cols):
                    M[r,c] += A[r,k]*B[k,c]
        return M

    def __rdiv__(A,x):
        """Computes x/A using Gauss-Jordan elimination where x is a scalar"""
        import copy
        n = A.cols
        if A.rows != n:
           raise ArithmeticError, "matrix not squared"
        indexes = range(n)
        A = copy.deepcopy(A)
        B = Matrix.identity(n,x)
        for c in indexes:
            for r in xrange(c+1,n):
                if abs(A[r,c])>abs(A[c,c]):
                    A.swap_rows(r,c)
                    B.swap_rows(r,c)
            p = 0.0 + A[c,c] # trick to make sure it is not integer
            for k in indexes:
                A[c,k] = A[c,k]/p
                B[c,k] = B[c,k]/p
            for r in range(0,c)+range(c+1,n):
                p = 0.0 + A[r,c] # trick to make sure it is not integer
                for k in indexes:
                    A[r,k] -= A[c,k]*p
                    B[r,k] -= B[c,k]*p
            # if DEBUG: print A, B
        return B

    def __div__(A,B):
        if isinstance(B,Matrix):
            return A*(1.0/B) # matrix/marix
        else:
            return (1.0/B)*A # matrix/scalar

    def swap_rows(A,i,j):
        for c in xrange(A.cols):
            A[i,c],A[j,c] = A[j,c],A[i,c]

    @property
    def t(A):
        """Transposed of A"""
        return Matrix(A.cols,A.rows, fill=lambda r,c: A[c,r])

def is_almost_symmetric(A, ap=1e-6, rp=1e-4):
    if A.rows != A.cols: return False
    for r in xrange(A.rows-1):
        for c in xrange(r):
            delta = abs(A[r,c]-A[c,r])
            if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
                return False
    return True

def is_almost_zero(A, ap=1e-6, rp=1e-4):
    for r in xrange(A.rows):
        for c in xrange(A.cols):
            delta = abs(A[r,c]-A[c,r])
            if delta>ap and delta>max(abs(A[r,c]),abs(A[c,r]))*rp:
                return False
    return True

def norm(A,p=1):
    if isinstance(A,(list,tuple)):
        return sum(x**p for x in A)**(1.0/p)
    elif isinstance(A,Matrix):
        if A.rows==1 or A.cols==1:
             return sum(norm(A[r,c])**p \
                for r in xrange(A.rows) \
                for c in xrange(A.cols))**(1.0/p)
        elif p==1:
             return max([sum(norm(A[r,c]) \
                for r in xrange(A.rows)) \
                for c in xrange(A.cols)])
        else:
             raise NotImplementedError
    else:
        return abs(A)

def condition_number(f,x=None,h=1e-6):
    if callable(f) and not x is None:
        return D(f,h)(x)*x/f(x)
    elif isinstance(f,Matrix): # if is the Matrix J
        return norm(f)*norm(1/f)
    else:
        raise NotImplementedError

def exp(x,ap=1e-6,rp=1e-4,ns=40):
    if isinstance(x,Matrix):
       t = s = Matrix.identity(x.cols)
       for k in range(1,ns):
           t = t*x/k   # next term
           s = s + t   # add next term
           if norm(t)<max(ap,norm(s)*rp): return s
       raise ArithmeticError, 'no convergence'
    elif type(x)==type(1j):
       return cmath.exp(x)
    else:
       return math.exp(x)

def Cholesky(A):
    import copy, math
    if not is_almost_symmetric(A):
        raise ArithmeticError, 'not symmetric'
    L = copy.deepcopy(A)
    for k in xrange(L.cols):
        if L[k,k]<=0:
            raise ArithmeticError, 'not positive definitive'
        p = L[k,k] = math.sqrt(L[k,k])
        for i in xrange(k+1,L.rows):
            L[i,k] /= p
        for j in xrange(k+1,L.rows):
            p=float(L[j,k])
            for i in xrange(k+1,L.rows):
                L[i,j] -= p*L[i,k]
    for  i in xrange(L.rows):
        for j in range(i+1,L.cols):
            L[i,j]=0
    return L

def is_positive_definite(A):
    if not is_symmetric(A):
        return False
    try:
        Cholesky(A)
        return True
    except RuntimeError:
        return False

def Markovitz(mu, A, r_free):
    """Assess Markovitz risk/return.
    Example:
    >>> cov = Matrix.from_list([[0.04, 0.006,0.02],
    ...                        [0.006,0.09, 0.06],
    ...                        [0.02, 0.06, 0.16]])
    >>> mu = Matrix.from_list([[0.10],[0.12],[0.15]])
    >>> r_free = 0.05
    >>> x, ret, risk = Markovitz(mu, cov, r_free)
    >>> print x
    [0.556634..., 0.275080..., 0.1682847...]
    >>> print ret, risk
    0.113915... 0.186747...
    """
    x = Matrix(A.rows, 1)
    print x
    x = (1/A)*(mu - r_free)
    print xrange
    x = x/sum(x[r,0] for r in range(x.rows))
    print x
    portfolio = [x[r,0] for r in range(x.rows)]
    portfolio_return = mu*x
    portfolio_risk = sqrt(x*(A*x))
    return portfolio, portfolio_return, portfolio_risk

def fit_least_squares(points, f):
    """
    Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
    where fitting_f(x[i]) is \sum_j c_j f[j](x[i])

    parameters:
    - a list of fitting functions
    - a list with points (x,y,dy)

    returns:
    - column vector with fitting coefficients
    - the chi2 for the fit
    - the fitting function as a lambda x: ....
    """
    def eval_fitting_function(f,c,x):
        if len(f)==1: return c*f[0](x)
        else: return sum(func(x)*c[i,0] for i,func in enumerate(f))
    A = Matrix(len(points),len(f))
    b = Matrix(len(points))
    for i in range(A.rows):
        weight = 1.0/points[i][2] if len(points[i])>2 else 1.0
        b[i,0] = weight*float(points[i][1])
        for j in range(A.cols):
            A[i,j] = weight*f[j](float(points[i][0]))
    c = (1.0/(A.t*A))*(A.t*b)
    chi = A*c-b
    chi2 = norm(chi,2)**2
    fitting_f = lambda x, c=c, f=f, q=eval_fitting_function: q(f,c,x)
    return c.data, chi2, fitting_f

# examples of fitting functions
def POLYNOMIAL(n):
    return [(lambda x, p=p: x**p) for p in range(n+1)]
CONSTANT  = POLYNOMIAL(0)
LINEAR    = POLYNOMIAL(1)
QUADRATIC = POLYNOMIAL(2)
CUBIC     = POLYNOMIAL(3)
QUARTIC   = POLYNOMIAL(4)

class Trader:
    def model(self,window):
        "the forecasting model"
        # we fit last few days quadratically
        points = [(x,y) for (x,y) in enumerate(window)]
        a,chi2,fitting_f = fit_least_squares(points,QUADRATIC)
        # and we extrapolate tomorrow's price
        price_tomorrow = fitting_f(len(points))
        return price_tomorrow

    def strategy(self,window):
        "the trading strategy"
        price_today = window[-1]
        price_tomorrow = self.model(window)
        if price_tomorrow>price_today:
            return 'buy'
        else:
            return 'sell'

    def simulate(self,data,cash=1000.0,shares=0.0,days=7,daily_rate=0.03/360):
        "find fitting parameters that optimize the trading strategy"
        for t in range(days,len(data)):
            window =  data[t-days:t]
            today_close = window[-1]
            suggestion = self.strategy(window)
            # and we buy or sell based on our strategy
            if cash>0 and suggestion=='buy':
                # we keep track of finances
                shares_bought = int(cash/today_close)
                shares += shares_bought
                cash -= shares_bought*today_close
            elif shares>0 and suggestion=='sell':
                cash += shares*today_close
                shares = 0.0
            # we assume money in the bank also gains an interest
            cash*=math.exp(daily_rate)
        # we return the net worth
        return cash+shares*data[-1]

def sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return cmath.sqrt(x)

def Jacobi_eigenvalues(A,checkpoint=False):
    """Returns U end e so that A=U*Matrix.diagonal(e)*transposed(U)
       where i-column of U contains the eigenvector corresponding to
       the eigenvalue e[i] of A.

       from http://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    """
    def maxind(M,k):
        j=k+1
        for i in xrange(k+2,M.cols):
            if abs(M[k,i])>abs(M[k,j]):
               j=i
        return j
    n = A.rows
    if n!=A.cols:
        raise ArithmeticError, 'matrix not squared'
    indexes = xrange(n)
    S = Matrix(n,n, fill=lambda r,c: float(A[r,c]))
    E = Matrix.identity(n)
    state = n
    ind = [maxind(S,k) for k in indexes]
    e = [S[k,k] for k in indexes]
    changed = [True for k in indexes]
    iteration = 0
    while state:
        if checkpoint: checkpoint('rotating vectors (%i) ...' % iteration)
        m=0
        for k in xrange(1,n-1):
            if abs(S[k,ind[k]])>abs(S[m,ind[m]]): m=k
            pass
        k,h = m,ind[m]
        p = S[k,h]
        y = (e[h]-e[k])/2
        t = abs(y)+sqrt(p*p+y*y)
        s = sqrt(p*p+t*t)
        c = t/s
        s = p/s
        t = p*p/t
        if y<0: s,t = -s,-t
        S[k,h] = 0
        y = e[k]
        e[k] = y-t
        if changed[k] and y==e[k]:
            changed[k],state = False,state-1
        elif (not changed[k]) and y!=e[k]:
            changed[k],state = True,state+1
        y = e[h]
        e[h] = y+t
        if changed[h] and y==e[h]:
            changed[h],state = False,state-1
        elif (not changed[h]) and y!=e[h]:
            changed[h],state = True,state+1
        for i in xrange(k):
            S[i,k],S[i,h] = c*S[i,k]-s*S[i,h],s*S[i,k]+c*S[i,h]
        for i in xrange(k+1,h):
            S[k,i],S[i,h] = c*S[k,i]-s*S[i,h],s*S[k,i]+c*S[i,h]
        for i in xrange(h+1,n):
            S[k,i],S[h,i] = c*S[k,i]-s*S[h,i],s*S[k,i]+c*S[h,i]
        for i in indexes:
            E[k,i],E[h,i] = c*E[k,i]-s*E[h,i],s*E[k,i]+c*E[h,i]
        ind[k],ind[h]=maxind(S,k),maxind(S,h)
        iteration+=1
    # sort vectors
    for i in xrange(1,n):
        j=i
        while j>0 and e[j-1]>e[j]:
            e[j],e[j-1] = e[j-1],e[j]
            E.swap_rows(j,j-1)
            j-=1
    # normalize vectors
    U = Matrix(n,n)
    for i in indexes:
        norm = sqrt(sum(E[i,j]**2 for j in indexes))
        for j in indexes: U[j,i] = E[i,j]/norm
    return U,e

def compute_correlation(stocks, key='arithmetic_return'):
    "The input must be a list of YStock(...).historical() data"
    # find trading days common to all stocks
    days = set()
    nstocks = len(stocks)
    iter_stocks = xrange(nstocks)
    for stock in stocks:
         if not days: days=set(x['date'] for x in stock)
         else: days=days.intersection(set(x['date'] for x in stock))
    n = len(days)
    v = []
    # filter out data for the other days
    for stock in stocks:
        v.append([x[key] for x in stock if x['date'] in days])
    # compute mean returns (skip first day, data not reliable)
    mus = [sum(v[i][k] for k in range(1,n))/n for i in iter_stocks]
    # fill in the covariance matrix
    var = [sum(v[i][k]**2 for k in range(1,n))/n - mus[i]**2 for i in iter_stocks]
    corr = Matrix(nstocks,nstocks,fill=lambda i,j: \
             (sum(v[i][k]*v[j][k] for k in range(1,n))/n - mus[i]*mus[j])/ \
             math.sqrt(var[i]*var[j]))
    return corr

def invert_minimum_residue(f,x,ap=1e-4,rp=1e-4,ns=200):
    import copy
    y = copy.copy(x)
    r = x-1.0*f(x)
    for k in xrange(ns):
        q = f(r)
        alpha = (q*r)/(q*q)
        y = y + alpha*r
        r = r - alpha*q
        residue = sqrt((r*r)/r.rows)
        if residue<max(ap,norm(y)*rp): return y
    raise ArithmeticError, 'no convergence'

def invert_bicgstab(f,x,ap=1e-4,rp=1e-4,ns=200):
    import copy
    y = copy.copy(x)
    r = x - 1.0*f(x)
    q = r
    p = 0.0
    s = 0.0
    rho_old = alpha = omega = 1.0
    for k in xrange(ns):
        rho = q*r
        beta = (rho/rho_old)*(alpha/omega)
        rho_old = rho
        p = beta*p + r - (beta*omega)*s
        s = f(p)
        alpha = rho/(q*s)
        r = r - alpha*s
        t = f(r)
        omega = (t*r)/(t*t)
        y = y + omega*r + alpha*p
        residue=sqrt((r*r)/r.rows)
        if residue<max(ap,norm(y)*rp): return y
    raise ArithmeticError, 'no convergence'

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

def solve_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    x = float(x) # make sure it is not int
    for k in xrange(ns):
        (fx, Dfx) = (f(x), D(f)(x))
        if norm(Dfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-fx/Dfx)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x
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

def optimize_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    x = float(x) # make sure it is not int
    for k in xrange(ns):
        (Dfx, DDfx) = (D(f)(x), DD(f)(x))
        if Dfx==0: return x
        if norm(DDfx) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-Dfx/DDfx)
        if norm(x-x_old)<max(ap,norm(x)*rp): return x
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

def partial(f,i,h=1e-4):
    def df(x,f=f,i=i,h=h):
        u = f([e+(h if i==j else 0) for j,e in enumerate(x)])
        w = f([e-(h if i==j else 0) for j,e in enumerate(x)])
        try:
            return (u-w)/2/h
        except TypeError:
            return [(u[i]-w[i])/2/h for i in range(len(u))]
    return df

def gradient(f, x, h=1e-4):
    return Matrix(len(x),fill=lambda r,c: partial(f,r,h)(x))

def hessian(f, x, h=1e-4):
    return Matrix(len(x),len(x),fill=lambda r,c: partial(partial(f,r,h),c,h)(x))

def jacobian(f, x, h=1e-4):
    partials = [partial(f,c,h)(x) for c in xrange(len(x))]
    return Matrix(len(partials[0]),len(x),fill=lambda r,c: partials[c][r])

def solve_newton_multi(f, x, ap=1e-6, rp=1e-4, ns=20):
    """
    Computes the root of a multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, solution of f(x)=0, as a list
    """
    x = Matrix.from_list([x]).t
    fx = Matrix.from_list([f(x.data)]).t
    for k in xrange(ns):
        (fx.data, J) = (f(x.data), jacobian(f,x.data))
        if norm(J) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-(1.0/J)*fx)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x.data
    raise ArithmeticError, 'no convergence'

def optimize_newton_multi(f, x, ap=1e-6, rp=1e-4, ns=20):
    """
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    """
    x = Matrix.from_list([x]).t
    for k in xrange(ns):
        (grad,H) = (gradient(f,x.data), hessian(f,x.data))
        if norm(H) < ap:
            raise ArithmeticError, 'unstable solution'
        (x_old, x) = (x, x-(1.0/H)*grad)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x.data
    raise ArithmeticError, 'no convergence'

def optimize_newton_multi_imporved(f, x, ap=1e-6, rp=1e-4, ns=20):
    """
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    """
    x = Matrix.from_list([x]).t
    fx = f(x.data)
    for k in xrange(ns):
        (grad,H) = (gradient(f,x.data), hessian(f,x.data))
        if norm(H) < ap:
            raise ArithmeticError, 'unstable solution'
        (fx_old, x_old, x) = (fx, x, x-(1.0/H)*grad)
        fx = f(x.data)
        if fx>fx_old:
            print k, x, grad.data
            (fx, x) = (fx_old, x_old)
            def g(y,f=f,x=x,grad=grad): return f((x+y*grad).data)
            y = optimize_newton(g,0,ap=0,rp=rp,ns=ns-k)
            x = x+y*grad
            fx = f(x.data)
        if k>2 and norm(x-x_old)<max(ap,norm(x)*rp): return x.data
    raise ArithmeticError, 'no convergence'

def fit(data, fs, b=None, ap=1e-6, rp=1e-4, ns=200, constraint=None):
    if not isinstance(fs,(list,tuple)):
        def g(b, data=data, f=fs, constraint=constraint):
            chi2 = sum(((y-f(b,x))/dy)**2 for (x,y,dy) in data)
            if constraint: chi2+=constraint(b)
            return chi2
        if isinstance(b,(list,tuple)):
            b = optimize_newton_multi_imporved(g,b,ap,rp,ns)
        else:
            b = optimize_newton(g,b,ap,rp,ns)
        return b, g(b,data,constraint=None)
    elif not b:
        a, chi2, ff = fit_least_squares(data, fs)
        return a, chi2
    else:
        na = len(fs)
        def core(b,data=data,fs=fs):
            A = Matrix.from_list([[fs[k](b,x)/dy for k in xrange(na)] \
                                  for (x,y,dy) in data])
            z = Matrix.from_list([[y/dy] for (x,y,dy) in data])
            a = (1/(A.t*A))*(A.t*z)
            chi2 = norm(A*a-z)**2
            return a.data, chi2
        def g(b,data=data,fs=fs,constraint=constraint):
            a, chi2 = core(b, data, fs)
            if constraint:
                chi += constraint(b)
            return chi2
        b = optimize_newton_multi_imporved(g,b,ap,rp,ns)
        a, chi2 = core(b,data,fs)
        return a+b,chi2

def integrate_naive(f, a, b, n=20):
    """
    Integrates function, f, from a to b using the trapezoidal rule
    >>> from math import sin
    >>> integrate(sin, 0, 2)
    1.416118...
    """
    a,b= float(a),float(b)
    h = (b-a)/n
    return h/2*(f(a)+f(b))+h*sum(f(a+h*i) for i in range(1,n))

def integrate(f, a, b, ap=1e-4, rp=1e-4, ns=20):
    """
    Integrates function, f, from a to b using the trapezoidal rule
    converges to precision
    """
    I = integrate_naive(f,a,b,1)
    for k in range(1,ns):
        I_old, I = I, integrate_naive(f,a,b,2**k)
        if k>2 and norm(I-I_old)<max(ap,norm(I)*rp): return I
    raise ArithmeticError, 'no convergence'

class QuadratureIntegrator:
    """
    Calculates the integral of the function f from points a to b
    using n Vandermonde weights and numerical quadrature.
    """
    def __init__(self,delta,order=4):
        h = float(delta)/(order-1)
        A = Matrix(order, order, fill = lambda r,c: (c*h)**r)
        s = Matrix(order, 1, fill = lambda r,c: (delta**(r+1))/(r+1))
        w = (1/A)*s
        self.packed = (h, order, w)
    def integrate(self,f,a):
        (h, order, w) = self.packed
        return sum(w[i,0]*f(a+i*h) for i in range(order))

def integrate_quadrature_naive(f,a,b,n=20,order=4):
    a,b = float(a),float(b)
    h = (b-a)/n
    q = QuadratureIntegrator((b-a)/n,order=order)
    return sum(q.integrate(f,a+i*h) for i in range(n))

class Cluster(object):
    def __init__(self,points,metric,weights=None):
        self.points, self.metric = points, metric
        self.k = len(points)
        self.w = weights or [1.0]*self.k
        self.q = dict((i,[i]) for i,e in enumerate(points))
        self.d = []
        for i in xrange(self.k):
            for j in xrange(i+1,self.k):
                m = metric(points[i],points[j])
                if not m is None:
                    self.d.append((m,i,j))
        self.d.sort()
        self.dd = []
    def parent(self,i):
        while isinstance(i,int): (parent, i) = (i, self.q[i])
        return parent, i
    def step(self):
        if self.k>1:
            # find new clusters to join
            (self.r,i,j),self.d = self.d[0],self.d[1:]
            # join them
            i,x = self.parent(i) # find members of cluster i
            j,y = self.parent(j) # find members if cluster j
            x += y               # join members
            self.q[j] = i        # make j cluster point to i
            self.k -= 1          # decrease cluster count
            # update all distances to new joined cluster
            new_d = [] # links not related to joined clusters
            old_d = {} # old links related to joined clusters
            for (r,h,k) in self.d:
                if h in (i,j):
                    a,b = old_d.get(k,(0.0,0.0))
                    old_d[k] = a+self.w[k]*r,b+self.w[k]
                elif k in (i,j):
                    a,b = old_d.get(h,(0.0,0.0))
                    old_d[h] = a+self.w[h]*r,b+self.w[h]
                else:
                    new_d.append((r,h,k))
            new_d += [(a/b,i,k) for k,(a,b) in old_d.items()]
            new_d.sort()
            self.d = new_d
            # update weight of new cluster
            self.w[i] = self.w[i]+self.w[j]
            # get new list of cluster memebrs
            self.v = [s for s in self.q.values() if isinstance(s,list)]
            self.dd.append((self.r,len(self.v)))
        return self.r, self.v

    def find(self,k):
        # if necessary start again
        if self.k<k: self.__init__(self.points,self.metric)
        # step until we get k clusters
        while self.k>k: self.step()
        # return list of cluster members
        return self.r, self.v

class NeuralNetwork:
    """
    Back-Propagation Neural Networks
    Placed in the public domain.
    Original author: Neil Schemenauer <nas@arctrix.com>
    Modified by: Massimo Di Pierro
    Read more: http://www.ibm.com/developerworks/library/l-neural/
    """

    @staticmethod
    def rand(a, b):
        """ calculate a random number where:  a <= rand < b """
        return (b-a)*random.random() + a

    @staticmethod
    def sigmoid(x):
        """ our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x) """
        return math.tanh(x)

    @staticmethod
    def dsigmoid(y):
        """ # derivative of our sigmoid function, in terms of the output (i.e. y) """
        return 1.0 - y**2

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = Matrix(self.ni, self.nh, fill=lambda r,c: self.rand(-0.2, 0.2))
        self.wo = Matrix(self.nh, self.no, fill=lambda r,c: self.rand(-2.0, 2.0))

        # last change in weights for momentum
        self.ci = Matrix(self.ni, self.nh)
        self.co = Matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError, 'wrong number of inputs'

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            s = sum(self.ai[i] * self.wi[i,j] for i in range(self.ni))
            self.ah[j] = self.sigmoid(s)

        # output activations
        for k in range(self.no):
            s = sum(self.ah[j] * self.wo[j,k] for j in range(self.nh))
            self.ao[k] = self.sigmoid(s)
        return self.ao[:]

    def back_propagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError, 'wrong number of target values'

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = self.dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = sum(output_deltas[k]*self.wo[j,k] for k in range(self.no))
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j,k] = self.wo[j,k] + N*change + M*self.co[j,k]
                self.co[j,k] = change
                #print N*change, M*self.co[j,k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i,j] = self.wi[i,j] + N*change + M*self.ci[i,j]
                self.ci[i,j] = change

        # calculate error
        error = sum(0.5*(targets[k]-self.ao[k])**2 for k in range(len(targets)))
        return error

    def test(self, patterns):
        for p in patterns:
            print p[0], '->', self.update(p[0])

    def weights(self):
        print 'Input weights:'
        for i in range(self.ni):
            print self.wi[i]
        print
        print 'Output weights:'
        for j in range(self.nh):
            print self.wo[j]

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, check=False):
        # N: learning rate
        # M: momentum factor
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, N, M)
            if check and i % 100 == 0:
                print 'error %-14f' % error


def test067():
    """
    >>> SP100 = ['AA', 'AAPL', 'ABT', 'AEP', 'ALL', 'AMGN', 'AMZN', 'AVP', 'AXP', 'BA', 'BAC', 'BAX', 'BHI', 'BK', 'BMY', 'BRK.B', 'CAT', 'C', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CPB', 'CSCO', 'CVS', 'CVX', 'DD', 'DELL', 'DIS', 'DOW', 'DVN', 'EMC', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'GD', 'GE', 'GILD', 'GOOG', 'GS', 'HAL', 'HD', 'HNZ', 'HON', 'HPQ', 'IBM', 'INTC', 'JNJ', 'JPM', 'KFT', 'KO', 'LMT', 'LOW', 'MA', 'MCD', 'MDT', 'MET', 'MMM', 'MO', 'MON', 'MRK', 'MS', 'MSFT', 'NKE', 'NOV', 'NSC', 'NWSA', 'NYX', 'ORCL', 'OXY', 'PEP', 'PFE', 'PG', 'PM', 'QCOM', 'RF', 'RTN', 'S', 'SLB', 'SLE', 'SO', 'T', 'TGT', 'TWX', 'TXN', 'UNH', 'UPS', 'USB', 'UTX', 'VZ', 'WAG', 'WFC', 'WMB', 'WMT', 'WY', 'XOM', 'XRX']
    >>> from datetime import date
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> for symbol in SP100:
    ...     key = symbol+'/2011'
    ...     if not key in storage:
    ...         storage[key] = YStock(symbol).historical(start=date(2011,1,1),
    ...                                                  stop=date(2011,12,31))
    
    """
    pass


def test068():
    """
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> appl = storage['AAPL/2011']
    >>> points = [(x,y['adjusted_close']) for (x,y) in enumerate(appl)]
    >>> draw(title='Apple Stock (2011)',xlab='trading day',ylab='adjusted close',
    ...      linesets = [{'label':'AAPL','data':points}],filename='images/aapl2011.png')
    
    """
    pass


def test069():
    """
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> appl = storage['AAPL/2011'][1:] # skip 1st day
    >>> points = [day['arithmetic_return'] for day in appl]
    >>> draw(title='Apple Stock (2011)',xlab='arithmetic return', ylab='frequency',
    ...      histsets = [{'data':points}],filename='images/aapl2011hist.png')
    
    """
    pass


def test070():
    """
    >>> from random import gauss
    >>> points = [(gauss(0,1),gauss(0,1),gauss(0,0.2),gauss(0,0.2)) for i in range(30)]
    >>> draw(title='example scatter plot', xrange=(-2,2), yrange=(-2,2),
    ...      ellisets = [{'data':points}],filename='images/scatter.png')
    
    """
    pass


def test071():
    """
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> points = []
    >>> for key in storage.keys('*/2011'):
    ...     v = [day['log_return'] for day in storage[key][1:]]
    ...     ret = sum(v)/len(v)
    ...     var = sum(x**2 for x in v)/len(v) - ret**2
    ...     points.append((var*math.sqrt(len(v)),ret*len(v),0.0002,0.02))
    >>> draw(title='S&P100 (2011)',xlab='risk',ylab='return',
    ...      ellisets = [{'data':points}],filename='images/sp100rr.png',
    ...      xrange = (min(p[0] for p in points),max(p[0] for p in points)),
    ...      yrange = (min(p[1] for p in points),max(p[1] for p in points)))
    
    """
    pass


def test072():
    """
    >>> def f(x,y): return (x-1)**2+(y-2)**2
    >>> points = [[f(0.1*i-3,0.1*j-3) for i in range(61)] for j in range(61)]
    >>> color2d(title='example 2d function',
    ...      data = points,filename='images/color2d.png')
    
    """
    pass


def test073():
    """
    >>> print fib(10)
    89
    
    """
    pass


def test074():
    """
    >>> walls, teared_down_walls = make_maze(n=20,d=2)
    
    """
    pass


def test075():
    """
    >>> vertices = range(10)
    >>> links = [(i,j,abs(math.sin(i+j+1))) for i in vertices for j in vertices]
    >>> graph = [vertices,links]
    >>> links = Dijkstra(graph,0)
    >>> for link in links: print link
    (1, 2, 0.897...)
    (2, 0, 0.141...)
    (3, 2, 0.420...)
    (4, 2, 0.798...)
    (5, 0, 0.279...)
    (6, 2, 0.553...)
    (7, 2, 0.685...)
    (8, 0, 0.412...)
    (9, 0, 0.544...)
    
    """
    pass


def test076():
    """
    >>> n,d = 4, 2
    >>> walls, links = make_maze(n,d)
    >>> symmetrized_links = [(i,j,1) for (i,j) in links]+[(j,i,1) for (i,j) in links]
    >>> graph = [range(n*n),symmetrized_links]
    >>> links = Dijkstra(graph,0)
    >>> paths = dict((i,(j,d)) for (i,j,d) in links)
    
    """
    pass


def test077():
    """
    >>> input = 'this is a nice day'
    >>> keys, encoded = encode_huffman(input)
    >>> print encoded
    10111001110010001100100011110010101100110100000011111111110
    >>> decoded = decode_huffman(keys,encoded)
    >>> print decoded == input
    True
    >>> print 1.0*len(input)/(len(encoded)/8)
    2.57...
    
    """
    pass


def test078():
    """
    >>> from math import log
    >>> input = 'this is a nice day'
    >>> w = [1.0*input.count(c)/len(input) for c in set(input)]
    >>> E = -sum(wi*log(wi,2) for wi in w)
    >>> print E
    3.23...
    
    """
    pass


def test079():
    """
    >>> dna1 = 'ATGCTTTAGAGGATGCGTAGATAGCTAAATAGCTCGCTAGA'
    >>> dna2 = 'GATAGGTACCACAATAATAAGGATAGCTCGCAAATCCTCGA'
    >>> print lcs(dna1,dna2)
    26
    
    """
    pass


def test080():
    """
    >>> bases = 'ATGC'
    >>> from random import choice
    >>> genes = [''.join(choice(bases) for k in range(10)) for i in range(20)]
    >>> chromosome1 = ''.join(choice(genes) for i in range(10))
    >>> chromosome2 = ''.join(choice(genes) for i in range(10))
    >>> z = needleman_wunsch(chromosome1, chromosome2)
    >>> color2d(title='Needleman-Wunsch', data=z,
    ...         filename='images/needleman.png')
    
    """
    pass


def test081():
    """
    >>> def f(x): return x*x-5.0*x
    >>> print f(0)
    0.0
    >>> f1 = D(f) # first derivative
    >>> print f1(0)
    -5.0
    >>> f2 = DD(f) # second derivative
    >>> print f2(0)
    2.00000...
    >>> f2 = D(f1) # second derivative
    >>> print f2(0)
    1.99999...
    
    """
    pass


def test082():
    """
    >>> X = [0.03*i for i in xrange(200)]
    >>> Y = {'label':'sin(x)','data':[(x,math.sin(x)) for x in X]}
    >>> Y1 = {'label':'Taylor 1st','data':[(x,x) for x in X[:100]]}
    >>> Y2 = {'label':'Taylor 3rd','data':[(x,x-x**3/6) for x in X[:100]]}
    >>> Y3 = {'label':'Taylor 5th','data':[(x,x-x**3/6+x**5/120) for x in X[:100]]}
    >>> draw(title='sin(x) approximations',filename='images/sin.png',linesets=[Y,Y1,Y2,Y3])
    
    """
    pass


def test083():
    """
    >>> a = math.pi/2
    >>> X = [0.03*i for i in xrange(200)]
    >>> Y = {'label':'sin(x)','data':[(x,math.sin(x)) for x in X]}
    >>> Y1 = {'label':'Taylor 2nd','data':[(x,1-(x-a)**2/2) for x in X[:150]]}
    >>> Y2 = {'label':'Taylor 4th','data':[(x,1-(x-a)**2/2+(x-a)**4/24) for x in X[:150]]}
    >>> Y3 = {'label':'Taylor 6th','data':[(x,1-(x-a)**2/2+(x-a)**4/24-(x-a)**6/720) for x in X[:150]]}
    >>> draw(title='sin(x) approximations',filename='images/sin2.png',linesets=[Y,Y1,Y2,Y2])
    
    """
    pass


def test084():
    """
    >>> for i in range(10):
    ...     x= 0.1*i
    ...     assert abs(myexp(x) - math.exp(x)) < 1e-4
    
    """
    pass


def test085():
    """
    >>> for i in range(10):
    ...     x= 0.1*i
    ...     assert abs(mysin(x) - math.sin(x)) < 1e-4
    
    """
    pass


def test086():
    """
    >>> for i in range(10):
    ...     x = 0.1*i
    ...     assert abs(mycos(x) - math.cos(x)) < 1e-4
    
    """
    pass


def test087():
    """
    >>> A = Matrix.from_list([[1.0,2.0],[3.0,4.0]])
    >>> print A + A      # calls A.__add__(A)
    [[2.0, 4.0], [6.0, 8.0]]
    >>> print A + 2      # calls A.__add__(2)
    [[3.0, 2.0], [3.0, 6.0]]
    >>> print A - 1      # calls A.__add__(1)
    [[0.0, 2.0], [3.0, 3.0]]
    >>> print -A         # calls A.__neg__()
    [[-1.0, -2.0], [-3.0, -4.0]]
    >>> print 5 - A      # calls A.__rsub__(5)
    [[4.0, -2.0], [-3.0, 1.0]]
    >>> b = Matrix.from_list([[1.0],[2.0],[3.0]])
    >>> print b + 2      # calls b.__add__(2)
    [[3.0], [4.0], [5.0]]
    
    """
    pass


def test088():
    """
    >>> A = Matrix.from_list([[1,2],[3,4]])
    >>> print A + 1j
    [[(1+1j), 2.0], [3.0, (4+1j)]]
    
    """
    pass


def test089():
    """
    >>> A = Matrix.from_list([[1.0,2.0],[3.0,4.0]])
    >>> print 2*A       # scalar * matrix
    [[2.0, 4.0], [6.0, 8.0]]
    >>> print A*A       # matrix * matrix
    [[7.0, 10.0], [15.0, 22.0]]
    >>> b = Matrix.from_list([[1],[2],[3]])
    >>> print b*b       # scalar product
    14
    
    """
    pass


def test090():
    """
    >>> points = [(math.cos(0.0628*t),math.sin(0.0628*t)) for t in range(200)]
    >>> points += [(0.02*t,0) for t in range(50)]
    >>> points += [(0,0.02*t) for t in range(50)]
    >>> draw(title='Linear Transformation',xlab='x',ylab='y',filename = 'la1.png',
    ...      ellisets = [{'data':points}], xrange=(-1,1), yrange=(-1,1))
    >>> def f(A,points,filename):
    ...      draw(title='Linear Transformation',xlab='x',ylab='y',filename=filename,
    ...           ellisets = [{'data':points},{'data':[A*x for x in points]}])
    >>> A1 = Matrix.from_list([[0.2,0],[0,1]])
    >>> f(A1, points, 'la2.png')
    >>> A2 = Matrix.from_list([[1,0],[0,0.2]])
    >>> f(A2, points, 'la3.png')
    >>> S = Matrix.from_list([[0.3,0],[0,0.3]])
    >>> f(S, points, 'la4.png')
    >>> s, c = math.sin(0.5), math.cos(0.5)
    >>> R = Matrix.from_list([[c,-s],[s,c]])
    >>> B1 = R*A1
    >>> f(B1, points, 'la5.png')
    >>> B2 = Matrix.from_list([[0.2,0.4],[0.5,0.3]])
    >>> f(B2, points, 'la6.png')
    
    """
    pass


def test091():
    """
    >>> A = Matrix.from_list([[1,2],[4,9]])
    >>> print 1/A
    [[9.0, -2.0], [-4.0, 1.0]]
    >>> print A/A
    [[1.0, 0.0], [0.0, 1.0]]
    >>> print A/2
    [[0.5, 1.0], [2.0, 4.5]]
    
    """
    pass


def test092():
    """
    >>> A = Matrix.from_list([[1,2],[3,4]])
    >>> print A.t
    [[1, 3], [2, 4]]
    
    """
    pass


def test093():
    """
    >>> A = Matrix.from_list([[1,2,2],[4,4,2],[4,6,4]])
    >>> b = Matrix.from_list([[3],[6],[10]])
    >>> x = (1/A)*b
    >>> print x
    [[-1.0], [3.0], [-1.0]]
    
    """
    pass


def test094():
    """
    >>> def f(x): return x*x-5.0*x
    >>> print condition_number(f,1)
    0.74999...
    >>> A = Matrix.from_list([[1,2],[3,4]])
    >>> print condition_number(A)
    21.0
    
    """
    pass


def test095():
    """
    >>> A = Matrix.from_list([[1,2],[3,4]])
    >>> print exp(A)
    [[51.96..., 74.73...], [112.10..., 164.07...]]
    
    """
    pass


def test096():
    """
    >>> A = Matrix.from_list([[4,2,1],[2,9,3],[1,3,16]])
    >>> L = Cholesky(A)
    >>> print is_almost_zero(A - L*L.t)
    True
    
    """
    pass


def test097():
    """
    >>> points = [(k,5+0.8*k+0.3*k*k+math.sin(k),2) for k in range(100)]
    >>> a,chi2,fitting_f = fit_least_squares(points,QUADRATIC)
    >>> for p in points[-10:]:
    ...     print p[0], round(p[1],2), round(fitting_f(p[0]),2)
    90 2507.89 2506.98
    91 2562.21 2562.08
    92 2617.02 2617.78
    93 2673.15 2674.08
    94 2730.75 2730.98
    95 2789.18 2788.48
    96 2847.58 2846.58
    97 2905.68 2905.28
    98 2964.03 2964.58
    99 3023.5 3024.48
    >>> draw(title='polynomial fit',xlab='t',ylab='e(t),o(t)',
    ...      pointsets=[{'label':'o(t)','data':points[:10]}],
    ...      linesets=[{'label':'e(t)','data':[(p[0],fitting_f(p[0])) for p in points[:10]]}],
    ...      filename = 'images/polynomialfit.png')
    
    """
    pass


def test098():
    """
    >>> from datetime import date
    >>> data = YStock.download('aapl','adjusted_close',
    ...        start=date(2011,1,1),stop=date(2011,12,31))
    >>> print Trader().simulate(data,cash=1000.0)
    1133.2463...
    >>> print 1000.0*math.exp(0.03)
    1030.4545...
    >>> print 1000.0*data[-1]/data[0]
    1228.8739...
    
    """
    pass


def test099():
    """
    >>> import random
    >>> A = Matrix(4,4)
    >>> for r in range(A.rows):
    ...     for c in range(r,A.cols):
    ...         A[r,c] = A[c,r] = random.gauss(10,10)
    >>> U,e = Jacobi_eigenvalues(A)
    >>> print is_almost_zero(U*Matrix.diagonal(e)*U.t-A)
    True
    
    """
    pass


def test100():
    """
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> symbols = storage.keys('*/2011')[:20]
    >>> stocks = [storage[symbol] for symbol in symbols]
    >>> corr = compute_correlation(stocks)
    >>> U,e = Jacobi_eigenvalues(corr)
    >>> draw(title='SP100 eigenvalues',xlab='i',ylab='e[i]',filename='images/sp100eigen.png',
    ...      linesets=[{'data':[(i,ei) for i,ei, in enumerate(e)]}])
    
    """
    pass


def test101():
    """
    >>> m = 30
    >>> x = Matrix(m,m,fill=lambda r,c:(r in(10,20) or c in(10,20))and 1. or 0.)
    >>> x.rows,x.cols = m*m, 1 # rearrange as Vector
    >>> def smear(x):
    ...     alpha, beta = 0.4, 8
    ...     for k in range(beta):
    ...        y = Matrix(x.rows,x.cols)
    ...        for r in range(m):
    ...            for c in range(m):
    ...                y[r*m+c,0] = (1.0-alpha/4)*x[r*m+c,0]
    ...                if c<m-1: y[r*m+c,0] += alpha * x[r*m+c+1,0]
    ...                if c>0:   y[r*m+c,0] += alpha * x[r*m+c-1,0]
    ...                if r<m-1: y[r*m+c,0] += alpha * x[r*m+c+m,0]
    ...                if c>0:   y[r*m+c,0] += alpha * x[r*m+c-m,0]
    ...        x = y
    ...     return y
    >>> y = smear(x)
    >>> z = invert_minimum_residue(smear,y,ns=1000)
    >>> y.rows, y.cols = m, m # rearrange as Matrix
    >>> z.rows, z.cols = m, m # rearrange as Matrix
    >>> color2d(title="Defocused image", data = y.as_list(),
    ...         filename='images/defocused.png')
    >>> color2d(title="refocus image", data = z.as_list(),
    ...         filename='images/refocused.png')
    
    """
    pass


def test102():
    """
    >>> def f(x): return (x-2)*(x-5)/10
    >>> print round(solve_fixed_point(f,1.0,rp=0),4)
    2.0
    
    """
    pass


def test103():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(solve_bisection(f,1.0,3.0),4)
    2.0
    
    """
    pass


def test104():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(solve_newton(f,1.0),4)
    2.0
    
    """
    pass


def test105():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(solve_secant(f,1.0),4)
    2.0
    
    """
    pass


def test106():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(solve_newton_stabilized(f,1.0,3.0),4)
    2.0
    
    """
    pass


def test107():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(optimize_bisection(f,2.0,5.0),4)
    3.5
    
    """
    pass


def test108():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(optimize_newton(f,3.0),3)
    3.5
    
    """
    pass


def test109():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(optimize_secant(f,3.0),3)
    3.5
    
    """
    pass


def test110():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(optimize_newton_stabilized(f,2.0,5.0),3)
    3.5
    
    """
    pass


def test111():
    """
    >>> def f(x): return (x-2)*(x-5)
    >>> print round(optimize_golden_search(f,2.0,5.0),3)
    3.5
    
    """
    pass


def test112():
    """
    >>> def f(x): return 2.0*x[0]+3.0*x[1]+5.0*x[1]*x[2]
    >>> df0 = partial(f,0)
    >>> df1 = partial(f,1)
    >>> df2 = partial(f,2)
    >>> x = (1,1,1)
    >>> print round(df0(x),4), round(df1(x),4), round(df2(x),4)
    2.0 8.0 5.0
    
    """
    pass


def test113():
    """
    >>> def f(x): return 2.0*x[0]+3.0*x[1]+5.0*x[1]*x[2]
    >>> print gradient(f, x=(1,1,1))
    [[1.999999...], [7.999999...], [4.999999...]]
    >>> print hessian(f, x=(1,1,1))
    [[0.0, 0.0, 0.0], [0.0, 0.0, 5.000000...], [0.0, 5.000000..., 0.0]]
    
    """
    pass


def test114():
    """
    >>> def f(x): return (2.0*x[0]+3.0*x[1]+5.0*x[1]*x[2], 2.0*x[0])
    >>> print jacobian(f, x=(1,1,1))
    [[1.9999999..., 7.999999..., 4.9999999...], [1.9999999..., 0.0, 0.0]]
    
    """
    pass


def test115():
    """
    >>> def f(x): return (x[0]+x[1], x[0]+x[1]**2-2)
    >>> print solve_newton_multi(f, x=(0,0))
    [1.0..., -1.0...]
    
    """
    pass


def test116():
    """
    >>> def f(x): return (x[0]-2)**2+(x[1]-3)**2
    >>> print optimize_newton_multi(f, x=(0,0))
    [2.0, 3.0]
    
    """
    pass


def test117():
    """
    >>> data = [(i, i+2.0*i**2+300.0/(i+10), 2.0) for i in range(1,10)]
    >>> fs = [(lambda b,x: x), (lambda b,x: x*x), (lambda b,x: 1.0/(x+b[0]))]
    >>> ab, chi2 = fit(data,fs,[5])
    >>> print ab, chi2
    [0.999..., 2.000..., 300.000..., 10.000...] ...
    
    """
    pass


def test118():
    """
    >>> from math import sin, cos
    >>> print integrate_naive(sin,0,3,n=2)
    1.6020...
    >>> print integrate_naive(sin,0,3,n=4)
    1.8958...
    >>> print integrate_naive(sin,0,3,n=8)
    1.9666...
    >>> print integrate(sin,0,3)
    1.9899...
    >>> print 1.0-cos(3)
    1.9899...
    
    """
    pass


def test119():
    """
    >>> from math import sin
    >>> print integrate_quadrature_naive(sin,0,3,n=2,order=2)
    1.60208248595
    >>> print integrate_quadrature_naive(sin,0,3,n=2,order=3)
    1.99373945223
    >>> print integrate_quadrature_naive(sin,0,3,n=2,order=4)
    1.99164529955
    
    """
    pass


def test120():
    """
    >>> def metric(a,b):
    ...     return math.sqrt(sum((x-b[i])**2 for i,x in enumerate(a)))
    >>> points = [[random.gauss(i % 5,0.3) for j in range(10)] for i in range(200)]
    >>> c = Cluster(points,metric)
    >>> r, clusters = c.find(1) # cluster all points until one cluster only
    >>> draw(title='clustering example',xlab='distance',ylab='number of clusters',
    ...      linesets = [{'data':c.dd[150:]}], filename = 'clustering1.png')
    >>> draw(title='clustering example (2d projection)',xlab='p[0]',ylab='p[1]',
    ...      ellisets = [{'data':[p[:2] for p in points]}],filename = 'clustering2.png')
    
    """
    pass


def test121():
    """
    >>> pat = [[[0,0], [0]], [[0,1], [1]], [[1,0], [1]], [[1,1], [0]]]
    >>> n = NeuralNetwork(2, 2, 1)
    >>> n.train(pat)
    >>> n.test(pat)
    [0, 0] -> [0.00...]
    [0, 1] -> [0.98...]
    [1, 0] -> [0.98...]
    [1, 1] -> [-0.00...]
    
    """
    pass


def test122():
    """
    >>> storage = PersistentDictionary('sp100.sqlite')
    >>> v = [day['arithmetic_return']*300 for day in storage['AAPL/2011'][1:]]
    >>> pat = [[v[i:i+5],[v[i+5]]] for i in range(len(v)-5)]
    >>> n = NeuralNetwork(5, 5, 1)
    >>> n.train(pat)
    >>> predictions = [n.update(item[0]) for item in pat]
    >>> success_rate = sum(1.0 for i,e in enumerate(predictions)
    ...                    if e[0]*v[i+5]>0)/len(pat)
    
    """
    pass

if __name__=='__main__':
    import os,doctest
    if not os.path.exists('images'): os.mkdir('images')
    doctest.testmod(optionflags=doctest.ELLIPSIS)