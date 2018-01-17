import numpy
import model
import json
import utils

with open('datasets/lfw/lfw_binary_attributes.json') as f: lfw=json.load(f)
with open('datasets/lfw/filelist.txt','r') as f: lfw_filelist=['images/'+x.strip() for x in f.readlines()]
def make_manifolds(a,s,im_path,t=[],visualize=False):
    '''
    a is the target attribute, s are exclusive attributes for the source,
    t are exclusive attributes for the target.
    '''
    S={k:set(v) for k,v in lfw['attribute_members'].items()}  #所有属性的集合 类型是字典
    T=lfw['attribute_gender_members']  #
    G=set(T[lfw['attribute_gender'][a]])  #G是个male的集合，因为a是Senior
    X=lfw_filelist.index(im_path)  #数字，索引

    def distfn(y,z):
        fy=[True if y in S[b] else False for b in sorted(S.keys())]
        fz=[True if z in S[b] else False for b in sorted(S.keys())]
        return sum(0 if yy==zz else 1 for yy,zz in zip(fy,fz))

    P=[i for i in range(len(lfw_filelist)) if i in G and i not in S[a] and not any(i in S[b] for b in t) and all(i in S[b] for b in s)]
    P= sorted([j for j in P if j!=X],key=lambda k: distfn(X,k))
    # target has correct gender, none of the source attributes and all of the target attributes
    Q=[i for i in range(len(lfw_filelist)) if i in G and i in S[a] and not any(i in S[b] for b in s) and all(i in S[b] for b in t)]
    Q= sorted([j for j in Q if j!=X and j not in P],key=lambda k: distfn(X,k))

    return [lfw_filelist[X]],[lfw_filelist[x] for x in P],[lfw_filelist[x] for x in Q]

if __name__=='__main__':

    K = 100
    max_iter = 500
    delta_list = [0.4]
    color_postprocess = True

    test_image_paths=['images/lfw/Melchor_Cob_Castro/Melchor_Cob_Castro_0001.jpg',
                      'images/lfw/John_Stockton/John_Stockton_0001.jpg',
                      'images/lfw/Ralf_Schumacher/Ralf_Schumacher_0005.jpg',
                      'images/lfw/Charlton_Heston/Charlton_Heston_0002.jpg',
                      'images/lfw/Tom_Ridge/Tom_Ridge_0032.jpg',
                      'images/lfw/Silvio_Berlusconi/Silvio_Berlusconi_0023.jpg']

    attribute_pairs=[('Youth', 'Senior'), ('Mouth Closed', 'Mouth Slightly Open'),
                     ('Mouth Closed', 'Mouth Slightly Open'),('Narrow Eyes', 'Eyes Open'), ('Pale Skin', 'Flushed Face'),
                     ('Frowning', 'Smiling'),('No Beard', 'Mustache'), ('No Eyewear', 'Eyeglasses')]

    # load CUDA model
    vgg=model.vgg19g_DeepFeature()

    result=[]
    original=[]
    # for each test image
    for i, path in enumerate(test_image_paths):
        result.append([])
        im=utils.im_read(path)
        image_size=im.shape[:2]
        XF=vgg.get_Deep_Feature([im]) #求一个图片的list的平均的特征向量#TODO
        original.append(im)
        # for each transform
        for j, (a,b) in enumerate(attribute_pairs):
            _,P,Q=make_manifolds(b,[a],im_path=path)
            PF=vgg.get_Deep_Feature(utils.im_generator(P[:K], image_size))
            QF=vgg.get_Deep_Feature(utils.im_generator(Q[:K], image_size))
            if True:
                WF=(QF-PF)/((QF-PF)**2).mean()
            else:
                WF=(QF-PF)
            # for each interpolation step
            for delta in delta_list:
                print(path,b,delta)
                Y=vgg.Deep_Feature_inverse(XF + WF * delta, max_iter=max_iter, initial_image=im)
                result[-1].append(Y)

    result=numpy.asarray(result)
    original=numpy.asarray(original)
    if color_postprocess:
        result=utils.color_match(numpy.expand_dims(original,1),result)
    utils.im_write('results/demo1.png',m)
    print('Output is results/demo1.png')

