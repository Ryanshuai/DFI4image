import pickle #必须加，不然出现奇怪问题
import torch
import argparse
import match



if __name__=='__main__':

  # configure by command-line arguments
  parser=argparse.ArgumentParser(description='Generate high resolution face transformations.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--method',type=str,default='facehair', help='older,younger,facehair,female,eyeglasses')
  parser.add_argument('--input',type=str,default='tests/1.jpg',help='input color image')
  parser.add_argument('--K',type=int,default=100,help='number of nearest neighbors')
  config=parser.parse_args()

  # load models
  mt=match.Match()

  # Set the free parameters
  K=config.K
  X=config.input

  # classifier scores
  XA=mt.predict_scores([X])[0]
  #print(XA)

  # positive and negative constraints
  fields = mt.fields
  gender = fields.index('Male')
  smile = fields.index('Smiling')
  if config.method=='older':
    cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),True)]
    cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),False)]
  elif config.method=='younger':
    cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),False)]
    cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),True)]
  elif config.method=='facehair':
    cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('No_Beard'),True),(fields.index('Mustache'),False)]
    cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('No_Beard'),False),(fields.index('Mustache'),True)]
  elif config.method == 'eyeglasses':
    cP = [(gender, XA[gender] >= 0), (smile, XA[smile] >= 0), (fields.index('no_eyewear'), True)]
    cQ = [(gender, XA[gender] >= 0), (smile, XA[smile] >= 0), (fields.index('no_eyewear'), False)]
  elif config.method == 'female':
    cP = [(gender, XA[gender] >= 0), (smile, XA[smile] >= 0)]
    cQ = [(fields.index('Male'), XA[fields.index('Male')] < 0), (smile, XA[smile] >= 0)]
  else:
    raise ValueError('Unknown method')


  P,Q=mt.select_PQ(XA, cP, cQ, K)
  print(P)
  print(Q)
  print('ok')
