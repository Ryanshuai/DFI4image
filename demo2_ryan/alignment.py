import numpy




def fit_submanifold_landmarks_to_image(template,original,Xlm,face_d,face_p,landmarks=list(range(68))):

  lossX=numpy.empty((len(Xlm),),dtype=numpy.float64)
  MX=numpy.empty((len(Xlm),2,3),dtype=numpy.float64)
  nfail=0
  for i in range(len(Xlm)):
    lm=Xlm[i]
    try:
      M,loss=alignface.fit_face_landmarks(Xlm[i],template,landmarks=landmarks,image_dims=original.shape[:2])
      lossX[i]=loss
      MX[i]=M
    except alignface.FitError:
      lossX[i]=float('inf')
      MX[i]=0
      nfail+=1
  if nfail>1:
    print('fit submanifold, {} errors.'.format(nfail))
  a=numpy.argsort(lossX)
  return a,lossX,MX




def fit_face_landmarks(X,template,verbose=False,landmarks=[33,39,42,8],scale_landmarks=[39,42],location_landmark=33,image_dims=(400,400),twoscale=True):
  '''
  X is a N x 2 matrix of landmark coordinates in the frame of the original image
  X 是一个N*2的矩阵，记录了在原图中的 landmark 的坐标
  template is a N x 2 matrix of landmark coordinates in the frame of the template
  template是一个N*2的矩阵，记录了在template中的 landmark的坐标
  image_dims is the (H,W) of the template
  '''
  Xsl=X[scale_landmarks].T.astype(numpy.float64)
  Xll=X[location_landmark].astype(numpy.float64)
  X=numpy.concatenate([X[landmarks].T,numpy.ones((1,len(landmarks)))],axis=0)

  # setup loss function
  Y=template[landmarks].T
  if twoscale:
    def f(scale1,scale2,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],0.0,0.0,0.0]).reshape(2,3)
      J2=numpy.array([0.0,0.0,0.0,st,ct,delta[1]]).reshape(2,3)
      J3=numpy.array([-st*scale1,-ct*scale1,0.0,ct*scale2,-st*scale2,0.0]).reshape(2,3)
      J4=numpy.array([0.0,0.0,1.0*scale1,0.0,0.0,0.0]).reshape(2,3)
      J5=numpy.array([0.0,0.0,0.0,0.0,0.0,1.0*scale2]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum(),(MXmY*(J5.dot(X))).sum()])
      return loss,jac
    def g(scale1,scale2,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=numpy.array([ct*scale1,-st*scale1,delta[0]*scale1,st*scale2,ct*scale2,delta[1]*scale2]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss
  else:
    def f(scale,theta,delta,X):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      J1=numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      J2=scale*numpy.array([-st,-ct,0.0,ct,-st,0.0]).reshape(2,3)
      J3=scale*numpy.array([0.0,0.0,1.0,0.0,0.0,0.0]).reshape(2,3)
      J4=scale*numpy.array([0.0,0.0,0.0,0.0,0.0,1.0]).reshape(2,3)
      jac=numpy.array([(MXmY*(J1.dot(X))).sum(),(MXmY*(J2.dot(X))).sum(),(MXmY*(J3.dot(X))).sum(),(MXmY*(J4.dot(X))).sum()])
      return loss,jac
    def g(scale,theta,delta):
      ct=math.cos(theta)
      st=math.sin(theta)
      M=scale*numpy.array([ct,-st,delta[0],st,ct,delta[1]]).reshape(2,3)
      MXmY=(M.dot(X)-Y)
      loss=0.5*(MXmY**2).sum()
      return M,loss

  # scipy optimizer
  tsl=template[scale_landmarks]
  initial_scale=min(numpy.linalg.norm(tsl[0]-tsl[1])/(numpy.linalg.norm(Xsl[:,0]-Xsl[:,1])+1e-5),max(image_dims))
  initial_delta=template[location_landmark]/initial_scale-Xll
  if twoscale:
    x0=numpy.asarray([initial_scale,initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2],x0[3:5],*args)
    bounds=[(0,max(image_dims)),(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  else:
    x0=numpy.asarray([initial_scale,0.0,initial_delta[0],initial_delta[1]]).astype(numpy.float64)
    def opt_fn(x0,*args):
      return f(x0[0],x0[1],x0[2:4],*args)
    bounds=[(0,max(image_dims)),(-3.1415926,3.1415926),(-(max(image_dims)**2),max(image_dims)**2),(-(max(image_dims)**2),max(image_dims)**2)]
  #print('check gradient')
  #print('check_grad',scipy.optimize.check_grad(lambda x0,*args: opt_fn(x0,*args)[0],lambda x0,*args: opt_fn(x0,*args)[1],x0,X))
  if verbose: print('initial guess',x0)
  result=[]
  for method in ['L-BFGS-B','TNC']:
    result.append(scipy.optimize.minimize(opt_fn,x0,args=(X,),jac=True,method=method,bounds=bounds))
  if verbose: print('{} of {} methods converged.'.format(sum(1 for x in result if x.success),len(result)))
  if not any(x.success for x in result):
    raise FitError('Cannot align face to template.\n{}'.format(result))
    for x in result: print(x)
  result=argmin(result,lambda x: x.fun)
  if verbose: print(result)
  if twoscale:
    scale1=result.x[0]
    scale2=result.x[1]
    theta=result.x[2]
    delta=result.x[3:5]
    M,loss=g(scale1,scale2,theta,delta)
  else:
    scale=result.x[0]
    theta=result.x[1]
    delta=result.x[2:4]
    M,loss=g(scale,theta,delta)
  return M,loss
