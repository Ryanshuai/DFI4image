
import numpy
import deepmodel
import json
import utils

with open('datasets/lfw/lfw_binary_attributes.json') as f: lfw=json.load(f)
with open('datasets/lfw/filelist.txt','r') as f: lfw_filelist=['images/'+x.strip() for x in f.readlines()]
def make_manifolds(a,s=[],t=[],N=10,X=None,visualize=False):
    '''
    a is the target attribute, s are exclusive attributes for the source,
    t are exclusive attributes for the target.
    '''
    S={k:set(v) for k,v in lfw['attribute_members'].items()}
    T=lfw['attribute_gender_members']
    G=set(T[lfw['attribute_gender'][a]])
    if X is None:
        # test has correct gender, all of the source attributes and none of the target attributes
        X=[i for i in range(len(lfw_filelist)) if i in G and i not in S[a] and not any(i in S[b] for b in t) and all(i in S[b] for b in s)]
        random.seed(123)
        random.shuffle(X)
    else:
        X=[lfw_filelist.index(x.decode()) for x in X]

    def distfn(y,z):
        fy=[True if y in S[b] else False for b in sorted(S.keys())]
        fz=[True if z in S[b] else False for b in sorted(S.keys())]
        return sum(0 if yy==zz else 1 for yy,zz in zip(fy,fz))
    # source has correct gender, all of the source attributes and none of the target attributes
    # ranked by attribute distance to test image
    P=[i for i in range(len(lfw_filelist)) if i in G and i not in S[a] and not any(i in S[b] for b in t) and all(i in S[b] for b in s)]
    P=[sorted([j for j in P if j!=X[i]],key=lambda k: distfn(X[i],k)) for i in range(N)]
    # target has correct gender, none of the source attributes and all of the target attributes
    Q=[i for i in range(len(lfw_filelist)) if i in G and i in S[a] and not any(i in S[b] for b in s) and all(i in S[b] for b in t)]
    Q=[sorted([j for j in Q if j!=X[i] and j not in P[i]],key=lambda k: distfn(X[i],k)) for i in range(N)]

    return [lfw_filelist[x] for x in X],[[lfw_filelist[x] for x in y] for y in P],[[lfw_filelist[x] for x in y] for y in Q]

if __name__=='__main__':

    K = 100
    max_iter = 500
    delta_list = [0.4]
    color_postprocess = True

    # load CUDA model
    minimum_resolution=200
    model=deepmodel.vgg19g_torch(device_id=0)

    # read test data
    data=numpy.load('tests/dmt2-lfw-multiple-attribute-test.npz')
    pairs=list(data['pairs'][[0,1,2,4,5,6]]) # skip flushed face, not interesting
    X=data['X']


    # Set the free parameters
    # Note: for LFW, 0.4*8.82 is approximately equivalent to beta=0.4

    result=[]
    original=[]
    # for each test image
    for i in range(len(X)):
        result.append([])
        xX=X[i].decode().replace('lfw','lfw_aegan')
        o=utils.im_read(xX)
        image_dims=o.shape[:2]
        if min(image_dims)<minimum_resolution:
            s=float(minimum_resolution)/min(image_dims)
            image_dims=(int(round(image_dims[0]*s)),int(round(image_dims[1]*s)))
            o=utils.im_resize(o,image_dims)
        XF=model.get_Deep_Feature([o]) #求图片的平均的特征向量
        original.append(o)
        # for each transform
        for j,(a,b) in enumerate(pairs):
            a = a.decode()
            b = b.decode()
            _,P,Q=make_manifolds(b,[a],[],X=X[i:i+1],N=1)
            P=P[0]
            Q=Q[0]
            xP=[x.replace('lfw','lfw_aegan') for x in P]
            xQ=[x.replace('lfw','lfw_aegan') for x in Q]
            PF=model.get_Deep_Feature(utils.im_generator(xP[:K],image_dims))
            QF=model.get_Deep_Feature(utils.im_generator(xQ[:K],image_dims))
            if True:
                WF=(QF-PF)/((QF-PF)**2).mean()
            else:
                WF=(QF-PF)
            init=o
            # for each interpolation step
            for delta in delta_list:
                print(xX,b,delta)
                Y=model.Deep_Feature_inverse(XF+WF*delta,max_iter=max_iter,initial_image=init)
                result[-1].append(Y)
                max_iter=config.iter//2
                init=Y

    result=numpy.asarray(result)
    original=numpy.asarray(original)
    if color_postprocess:
        result=utils.color_match(numpy.expand_dims(original,1),result)
    m=utils.montage(numpy.concatenate([numpy.expand_dims(original,1),result],axis=1))
    utils.im_write('results/demo1.png',m)
    print('Output is results/demo1.png')

