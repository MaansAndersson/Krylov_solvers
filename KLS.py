# Måns Andersson 2020
from dolfin import *
import time 

import cupy as cp
import numpy as np 
import scipy as sp
from scipy.sparse import csr_matrix

def linear_solver(A, u, b, M = None, solver = 'CG', TOL=1e-9, MaxIt = 10000, timing = False, solver_info = False):
  if timing:
    start = time.clock()
  mat = as_backend_type(A).mat()
  ai, aj, av = mat.getValuesCSR()
  Acsr = sp.sparse.csr_matrix((av, aj, ai))

  #Transport to GPU
  bGpu = cp.asarray(b[:])
  Agpu = cp.sparse.csr_matrix(Acsr)
  dia = cp.asarray(Acsr.diagonal()[:]) 

  if isinstance(u, str):
    if str == 'b':
      uu = cp.zeros_like(b, dtype=np.float64)
    else: 
      uu = cp.zeros_like(b, dtype=np.float64)  
  else:
    uu = cp.asarray(u[:])

  if solver == 'CG':
    x = fit_CG(Agpu, bGpu, uu, TOL, MaxIt, solver_info).get()
  elif solver == 'Jacobi':
    x = fit_Jacobi(Agpu, bGpu, uu, dia, TOL, MaxIt, solver_info).get()
  elif solver == 'BCG':
    x = fit_BCG(Agpu, bGpu, uu, TOL, MaxIt, solver_info).get()
  elif solver == 'CG' and M:
    mat = as_backend_type(M).mat()
    mi, mj, mv = mat.getValuesCSR()
    Mcsr = sp.sparse.csr_matrix((mv, mj, mi))
    x = fit_CG(Agpu, bGpu, uu, Mcsr, TOL, MaxIt, solver_info).get()
  else: # CGNS
    x = fit_prec_csrmv(Agpu, bGpu, uu, TOL, MaxIt, solver_info).get()

  if timing:
    end = time.clock()
    print("   Linear solver timing", end - start)
  return x

# CG
def fit_CG(A, b, x, tol, max_iter, solver_info):
    xp = cp.get_array_module(A) 
    #x = cp.zeros_like(b, dtype=np.float64) 
    r0 = b - A.dot(x)
    p = r0
    for i in range(max_iter):
        #Ap = A.dot(p)
        a = xp.inner(r0, r0) / xp.inner(p, A.dot(p))
        x += a * p
        r1 = r0 - a * A.dot(p)
        #print(xp.linalg.norm(r1))
        if xp.linalg.norm(r1) < tol:
            if solver_info:
              print('   Linear solver residual norm; ',xp.linalg.norm(r1), 'n of iterations: ', i)
            return x
        b = xp.inner(r1, r1) / xp.inner(r0, r0)
        p = r1 + b * p
        r0 = r1
    print('   Linear solver failed to converge. Increase max-iter or tol. Current rtol', xp.linalg.norm(r1))
    return x

def fit_BCG(A, b, x, tol, max_iter, solver_info):
    xp = cp.get_array_module(A) 
    alpha = 1.
    beta = 0.
    
    xh = cp.zeros_like(b, dtype=np.float64) 
    r0 = b - A.dot(x)
    r0h = b - cp.cusparse.csrmv(A, x, alpha = alpha, beta = beta, transa = True)
    p = r0
    ph = r0h
    for i in range(max_iter):
        #Ap = A.dot(p)
        a = xp.inner(r0h, r0) / xp.inner(ph, A.dot(p))
        x += a * p
        xh += a * ph
        r1 = r0 - a * A.dot(p)
        r1h = r0h - a * cp.cusparse.csrmv(A,ph, alpha = alpha, beta = beta, transa = True)
        #print(xp.linalg.norm(r1))
        if xp.linalg.norm(r1) < tol:
            if solver_info :
              print('   Linear solver residual norm; ',xp.linalg.norm(r1), xp.linalg.norm(r1h), 'n of iterations: ', i)
            return x
        b = xp.inner(r1h, r1) / xp.inner(r0h, r0)
        p = r1 + b * p
        ph = r1h + b * ph
        r0 = r1
        r0h = r1h
    print('   Linear solver failed to converge. Increase max-iter or tol. Current rtol', xp.linalg.norm(r1), xp.linalg.norm(r1h))
    return x

def fit_prec_csrmv(A, b, x, tol, max_iter, solver_info):
    xp = cp.get_array_module(A)
    #x = xp.zeros_like(b, dtype=np.float64) 
    alpha = 1.
    beta = 0.
    
    r0 = b - A.dot(x)
    z0 = cp.cusparse.csrmv(A, r0, alpha = alpha, beta = beta, transa = True)
    p = z0
    for i in range(max_iter):

        a = xp.inner(z0, r0) / xp.inner(p, A.dot(p))

        x += a * p
        r1 = r0 - a * A.dot(p)
        z1 = cp.cusparse.csrmv(A,r1, alpha = alpha, beta = beta, transa = True) 

        if cp.linalg.norm(r1) < tol:
            if solver_info :
              print('   Linear solver residual norm; ',xp.linalg.norm(r1), 'n of iterations: ', i)
            return x
        b = xp.inner(r1, z1) / xp.inner(r0, z0)
        p = z1 + b * p
        r0 = r1
        z0 = z1
    print('   Linear solver failed to converge. Increase max-iter or tol. Current rtol', xp.linalg.norm(r1))
    return x
  
def fit_prec(A, b, x, M, tol, max_iter, solver_info):
    xp = cp.get_array_module(A)
    #x = xp.zeros_like(b, dtype=np.float64) 
    alpha = 1.
    beta = 0.
    
    r0 = b - A.dot(x)
    z0 = cp.dot(M, r0)
    p = z0
    for i in range(max_iter):

        a = xp.inner(z0, r0) / xp.inner(p, A.dot(p))

        x += a * p
        r1 = r0 - a * A.dot(p)
        z1 = cp.dot(M,r1)

        if cp.linalg.norm(r1) < tol:
            if solver_info :
              print('   Linear solver residual norm; ',xp.linalg.norm(r1), 'n of iterations: ', i)
            return x
        b = xp.inner(r1, z1) / xp.inner(r0, z0)
        p = z1 + b * p
        r0 = r1
        z0 = z1
    print('   Linear solver failed to converge. Increase max-iter or tol. Current rtol', xp.linalg.norm(r1))
    return x

def fit_Jacobi(A, b, x_init, dia, tol, max_iter, solver_info):
  xp = cp.get_array_module(A) 
  #x_init = cp.zeros_like(b, dtype=np.float64) 
  #x_1 = 1/D * (b - (A-D)*x_0)
  D = xp.sparse.dia_matrix((dia, xp.array([0], 'i')), shape = A.shape)
  LU = A - D
  x = x_init
  D_inv = xp.sparse.dia_matrix((1 / dia, xp.array([0], 'i')),  shape = A.shape)
  for i in range(max_iter):
      x_new = D_inv.dot( b - LU.dot(x))
      if xp.linalg.norm(x_new - x) < tol:
        if solver_info:
          print(xp.linalg.norm(x_new - x))
        return x_new
      x = x_new
  print('   Linear solver failed to converge. Increase max-iter or tol. Current rtol', xp.linalg.norm(x_new - x))
  return x

def project_(v, V):
  z = TestFunction(V)
  w = TrialFunction(V)
  projection = Function(V)

  b = assemble(inner(z,v)*dx) 
  A = assemble(inner(w,z)*dx)

  projection.vector()[:] = linear_solver(A, projection.vector()[:], b, 'BCG')
  return projection
