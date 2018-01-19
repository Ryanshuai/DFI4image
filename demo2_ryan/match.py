import pickle
import lutorpy # 必须加 !!!!, 不然提示没有require
import numpy
import math
import os
import alignment
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
require('torch')
require('cunn')
require('cudnn')


fields = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young',
    'asian',
    'baby',
    'black',
    'brown_eyes',
    'child',
    'color_photo',
    # 'eyeglasses2',
    'eyes_open',
    'flash',
    'flushed_face',
    'frowning',
    'fully_visible_forehead',
    'harsh_lighting',
    # 'high_cheekbones2',
    'indian',
    'middle_aged',
    'mouth_closed',
    # 'mouth_slightly_open2',
    'mouth_wide_open',
    'no_eyewear',
    'obstructed_forehead',
    'outdoor',
    'partially_visible_forehead',
    'posed_photo',
    'round_face',
    'round_jaw',
    'senior',
    'shiny_skin',
    'soft_lighting',
    'square_face',
    'strong_nose_mouth_lines',
    'sunglasses',
    'teeth_not_visible',
    'white',
]

class Attributes_Classifier:
    def __init__(self, file='models/facemodel/'):
        '''
        This model takes a 160x160 input which is a single face aligned to
        self.celeba_template (which is a set of 68 dlib landmarks). Call
        preprocess() then forward() to get a 1024-dim feature vector. Then
        call predict_attributes() on the features to get 40 attribute
        decision values.
        '''
        self.clf = {}
        self.model = torch.load(file + 'model.t7')
        self.model._evaluate()  # set model to test mode
        self.fields = tuple(fields)
        self.meanstd = {'mean': numpy.array([0.485, 0.456, 0.406]), 'std': numpy.array([0.229, 0.224, 0.225])}
        self.celeba_template = numpy.load(file + 'celeba_dlib_template.npz')['template']
        self.clf = {k: pickle.load(open(file+'classifiers/{}.pkl'.format(k), 'rb'), encoding='latin') for k in fields}
    def preprocess(self, orig_X):
        '''X is an N * H * W * 3 set of images in the range [0, 1].'''
        X = orig_X - self.meanstd['mean']
        X = X / self.meanstd['std']
        def center_crop(X, crop):
            '''X is [N, H, W, 3]'''
            h1 = int(math.ceil((X.shape[1] - crop) / 2))
            w1 = int(math.ceil((X.shape[2] - crop) / 2))
            return X[:, h1:h1 + crop, w1:w1 + crop, :]
        pre_X = center_crop(X, 160)
        return pre_X

    def predict_attributes(self, pre_X):
        '''X is an N x H x W x 3 set of images'''
        thX = torch.fromNumpyArray(pre_X.transpose(0, 3, 1, 2))
        cuthX = thX._cuda()
        # assert torch.typename(cuthX) in cudnn.typemap.keys()
        thY = self.model._forward(cuthX)
        feature = thY.asNumpyArray()
        scores = numpy.array([self.clf[k].decision_function(feature) for k in fields]).T
        return scores


class Match():
    def __init__(self):
        # used to predict_scores
        # 加载多次 !!!, 可传入此参数
        self.face_d, self.face_p = alignment.load_face_detector(
            predictor_path='models/facemodel/shape_predictor_68_face_landmarks.dat')
        self.attr_clf = Attributes_Classifier()
        # used to select_PQ
        data = numpy.load('datasets/facemodel/attributes.npz')
        self.fields = fields
        self.filelist = tuple(data['filelist'])
        self._scores = data['scores']
        self._binary_score = (self._scores >= 0)
        # mark 70% as confident
        self._confident = numpy.zeros_like(self._scores, dtype=numpy.bool)
        self._confident[self._scores > numpy.percentile(self._scores[self._scores >= 0], 30, axis=0)] = True
        self._confident[self._scores < numpy.percentile(self._scores[self._scores < 0], 70, axis=0)] = True

        # only attributes related to faces for nearest neighbor distance
        self.distance_idx = [fields.index(x) for x in
                             ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                              'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                              'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                              'Narrow_Eyes', 'No_Beard', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Sideburns',
                              'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Lipstick', 'Young', 'asian', 'baby',
                              'black', 'brown_eyes', 'child', 'eyes_open', 'frowning', 'fully_visible_forehead',
                              'indian', 'middle_aged', 'mouth_closed', 'mouth_wide_open', 'no_eyewear',
                              'obstructed_forehead', 'partially_visible_forehead', 'senior', 'shiny_skin',
                              'strong_nose_mouth_lines', 'sunglasses', 'teeth_not_visible', 'white']]

    def predict_scores(self, img_path):
        """
        :param img_path: [path_str_list],如[tests/1.jpg]
        :return: [N, c], scores
        """
        scores=[]
        for ipath in img_path:
          # 特征点重复检测了，可传入此参数!!!!!!
          lm,x=alignment.detect_landmarks(ipath,self.face_d,self.face_p)
          M,loss=alignment.fit_face_landmarks(lm,self.attr_clf.celeba_template,image_dims=[160,160],twoscale=False)
          Y=alignment.warp_to_template(x,M,image_dims=[160,160])
          Y=numpy.expand_dims(Y,0)
          Y=self.attr_clf.preprocess(Y)
          scr=self.attr_clf.predict_attributes(Y)
          scores.append(scr[0])
        scores=numpy.asarray(scores)
        return scores


    def select(self, constraints, attributes):
        def admissible(i):
            return all(self._binary_score[i, j] == v and self._confident[i, j] for j, v in constraints)

        S = numpy.asarray([i for i in range(len(self.filelist)) if admissible(i)])
        if len(S) < 1: return []
        # cdist : metric距离度量 (minkowski,1) 1范数
        knn = numpy.argsort(
            scipy.spatial.distance.cdist(numpy.expand_dims(attributes[self.distance_idx], 0),
                                         self._binary_score[S][:, self.distance_idx], 'minkowski', 1) # 连续和离散？？
        ).astype(numpy.int32)
        return [self.filelist[i] for i in S[knn[0]]]

    def select_PQ(self, attr_score, constraints_P, constraints_Q, K=100, redundancy = 4):
        """
        :param attr_score: [c,]
        :param  constraints_P, constraints_Q:例如
                                 cP=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),True)]
                                 cQ=[(gender,XA[gender]>=0),(smile,XA[smile]>=0),(fields.index('Young'),False)]
        :param K: knn个数

        :return: [P_index], [Q_index]
                (文件名列表, 返回 4*k, 不足4*K全部返回)
        """
        P = self.select(constraints_P, attr_score)
        if len(P) > redundancy * K:
            P = P[:K]
        Q = self.select(constraints_Q, attr_score)
        if len(Q) > redundancy * K:
            Q = Q[:K]
        return P, Q




