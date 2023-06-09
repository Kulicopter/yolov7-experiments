import time
import cv2
import numpy as np
import uuid

class DetectRect:
    def __init__(self, tl: np.array, br: np.array, tag, score):
        self.tl = tl
        self.br = br
        self.tag = tag
        self.score= score

    # Верхняя левая точка детектирования
    tl = np.zeros((2))
    # Правая нижняя точка детектирования
    br = np.zeros((2))
    # Тэг класса детектирования
    tag = -1
    # Скор детектирования
    score = -1.0

class TrackObject:
    def __init__(self, rect: DetectRect):
        self.id = uuid.uuid4()
        self.trackHistory = [rect]
        self.saturation = 1
        self.ttl = 10

    # Идентификатор трека
    id = None
    # История детектирований трека
    trackHistory = []
    # Насыщение валидации
    saturation = 0
    # TTL
    ttl = 10

    def last(self):
        if len(self.trackHistory) > 0:
            return self.trackHistory[len(self.trackHistory) - 1]
        return None

    def addRect(self, rect: DetectRect):
        self.trackHistory.append(rect)
        self.saturation += 1

    def dialecticScore_percentile(self, batch: int):
        batchSize = np.minimum(batch, len(self.trackHistory))
        dialecticBatch = []
        k = len(self.trackHistory) - batchSize
        for i in range(batchSize):
            score = self.trackHistory[i + k].score
            if self.trackHistory[i + k].tag:
                score *= -1
            score = (score + 1) * 0.5
            dialecticBatch.append(score)

        dialecticScore = 0
        if (len(dialecticBatch)):
            dialecticScore = \
                np.percentile(np.asarray(dialecticBatch), 40)

        return dialecticScore

def IoU(rect1: DetectRect, rect2: DetectRect):
    i_tl = np.asarray([np.maximum(rect1.tl[0], rect2.tl[0]),
                       np.maximum(rect1.tl[1], rect2.tl[1])])
    i_br = np.asarray([np.minimum(rect1.br[0], rect2.br[0]),
                       np.minimum(rect1.br[1], rect2.br[1])])

    i_dx = i_br[0] - i_tl[0]
    i_dy = i_br[1] - i_tl[1]
    if i_dx < 0 or i_dy < 0:
        return 0

    i_S = (i_dx) * (i_dy)
    S1 = (rect1.br[0] - rect1.tl[0]) * \
         (rect1.br[1] - rect1.tl[1])
    S2 = (rect2.br[0] - rect2.tl[0]) * \
         (rect2.br[1] - rect2.tl[1])

    return float(i_S) / float(S1 + S2 - i_S)




