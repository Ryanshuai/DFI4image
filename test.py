
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

    test_image_paths=['images/lfw_aegan/Melchor_Cob_Castro/Melchor_Cob_Castro_0001.jpg',
                      'images/lfw_aegan/John_Stockton/John_Stockton_0001.jpg',
                      'images/lfw_aegan/Ralf_Schumacher/Ralf_Schumacher_0005.jpg',
                      'images/lfw_aegan/Charlton_Heston/Charlton_Heston_0002.jpg',
                      'images/lfw_aegan/Tom_Ridge/Tom_Ridge_0032.jpg',
                      'images/lfw_aegan/Silvio_Berlusconi/Silvio_Berlusconi_0023.jpg']

    attribute_pairs=[('Youth', 'Senior'), ('Mouth Closed', 'Mouth Slightly Open'),
                     ('Mouth Closed', 'Mouth Slightly Open'),('Narrow Eyes', 'Eyes Open'), ('Pale Skin', 'Flushed Face'),
                     ('Frowning', 'Smiling'),('No Beard', 'Mustache'), ('No Eyewear', 'Eyeglasses')]

    # load CUDA model
    model=deepmodel.vgg19g_torch()

    result=[]
    original=[]
    # for each test image
    for path in test_image_paths:
        result.append([])
        im=utils.im_read(path)
        image_size=im.shape[:2]
        XF=model.get_Deep_Feature([im]) #求图片的平均的特征向量#TODO
        original.append(im)
        # for each transform
        for j,(a,b) in enumerate(attribute_pairs):
            _,P,Q=make_manifolds(b,[a],[],X=path,N=1)
            P=P[0]
            Q=Q[0]
            xP=[x.replace('lfw','lfw_aegan') for x in P]
            xQ=[x.replace('lfw','lfw_aegan') for x in Q]
            PF=model.get_Deep_Feature(utils.im_generator(xP[:K],image_size))
            QF=model.get_Deep_Feature(utils.im_generator(xQ[:K],image_size))
            if True:
                WF=(QF-PF)/((QF-PF)**2).mean()
            else:
                WF=(QF-PF)
            # for each interpolation step
            for delta in delta_list:
                print(path,b,delta)
                Y=model.Deep_Feature_inverse(XF+WF*delta,max_iter=max_iter,initial_image=im)
                result[-1].append(Y)

    result=numpy.asarray(result)
    original=numpy.asarray(original)
    if color_postprocess:
        result=utils.color_match(numpy.expand_dims(original,1),result)
    m=utils.montage(numpy.concatenate([numpy.expand_dims(original,1),result],axis=1))
    utils.im_write('results/demo1.png',m)
    print('Output is results/demo1.png')

