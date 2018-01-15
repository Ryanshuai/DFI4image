
import numpy
import os.path
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont



def read(ipath,dtype=numpy.float32):
    '''
    Returns a H x W x 3 RGB image in the range of [0,1].
    '''
    img=PIL.Image.open(ipath)
    if img.mode!='RGB':
        img=img.convert('RGB')
    return numpy.asarray(img,dtype=dtype)/255.0

def write(opath,I,**kwargs):
    '''
    Given a H x W x 3 RGB image it is clipped to the range [0,1] and
    written to an 8-bit image file.
    '''
    img=PIL.Image.fromarray((I*255).clip(0,255).astype(numpy.uint8))
    ext=os.path.splitext(opath)[1]
    if ext=='.jpg':
        quality=kwargs['quality'] if 'quality' in kwargs else 95
        img.save(opath,quality=quality,optimize=True)
    elif ext=='.png':
        img.save(opath)
    else:
        # I do not want to save unknown extensions because there is no
        # expectation that the default save options are reasonable.
        raise ValueError('Unknown image extension ({})'.format(ext))

def resize(I,shape):
    image=PIL.Image.fromarray((I*255).clip(0,255).astype(numpy.uint8))
    if shape[0]<I.shape[0] and shape[1]<I.shape[1]:
        image=image.resize((shape[1],shape[0]),PIL.Image.BICUBIC)
    else:
        image=image.resize((shape[1],shape[0]),PIL.Image.LANCZOS)
    return numpy.asarray(image)/255.0

def montage(M,sep=0,canvas_value=0):
    # row X col X H X W X C
    assert M.ndim==5
    canvas=numpy.ones((M.shape[0]*M.shape[2]+(M.shape[0]-1)*sep,M.shape[1]*M.shape[3]+(M.shape[1]-1)*sep,M.shape[4]),dtype=M.dtype)*canvas_value
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            canvas[i*(M.shape[2]+sep):i*(M.shape[2]+sep)+M.shape[2],j*(M.shape[3]+sep):j*(M.shape[3]+sep)+M.shape[3]]=M[i,j]
    return canvas

def concatenate(X,axis,canvas_value=0,gravity=(-1)):
    '''
    Given a sequence of images, concatenate them along the given axis,
    expanding the other axes as needed. If gravity is zero then the original
    data will be centered in the output domain. Negative or positive gravity
    will cause it to be flush with the lower or upper bound, respectively.
    '''
    outshape=[sum(x.shape[i] for x in X) if i==axis else max(x.shape[i] for x in X) for i in range(X[0].ndim)]
    Y=[]
    for x in X:
        newshape=list(outshape)
        newshape[axis]=x.shape[axis]
        if gravity>0:
            Y.append(numpy.pad(x,[(newshape[i]-x.shape[i],0) for i in range(x.ndim)],'constant',constant_values=canvas_value))
        elif gravity==0:
            Y.append(numpy.pad(x,[((newshape[i]-x.shape[i])//2,(newshape[i]-x.shape[i])-(newshape[i]-x.shape[i])//2) for i in range(x.ndim)],'constant',constant_values=canvas_value))
        else:
            Y.append(numpy.pad(x,[(0,newshape[i]-x.shape[i]) for i in range(x.ndim)],'constant',constant_values=canvas_value))
    return numpy.concatenate(Y,axis=axis)

def image_feed(S,image_dims):
    '''
    Given a list of file paths and a 2-tuple of (H, W), yields H x W x 3 images.
    '''
    for x in S:
        I=read(x)
        if I.shape[:2]!=image_dims:
            yield resize(I,tuple(image_dims))
        else:
            yield I

def color_match(A,B):
    '''
    A is a rank 5 tensor (column of original images)
    B is a rank 5 tensor (grid of images)
    '''
    A=numpy.asarray(A)
    B=numpy.asarray(B)
    print('Computing color match',A.shape,B.shape)
    m=A.reshape(A.shape[0],1,-1).mean(axis=2)
    m=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(m,-1),-1),-1)
    s=(A-m).reshape(A.shape[0],1,-1).std(axis=2)
    s=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(s,-1),-1),-1)
    m2=B.reshape(B.shape[0],B.shape[1],-1).mean(axis=2)
    m2=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(m2,-1),-1),-1)
    s2=(B-m2).reshape(B.shape[0],B.shape[1],-1).std(axis=2)
    s2=numpy.expand_dims(numpy.expand_dims(numpy.expand_dims(s2,-1),-1),-1)
    return (B-m2)*(s+1e-8)/(s2+1e-8)+m

