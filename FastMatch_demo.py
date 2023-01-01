import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def FastMatch(template, img):
    epsilon                 = 0.15
    delta                   = 0.25
    photometricInvariance   = 0
    templateMask            = np.ones(template.shape)

    img          = MakeOdd(img)
    template     = MakeOdd(template)
    templateMask = MakeOdd(templateMask)
    
    ## image dimisions
    h1, w1 = template.shape[:2]
    h2, w2 = img.shape[:2]
    r1x    = 0.5 * (w1 - 1)
    r1y    = 0.5 * (h1 - 1)
    r2x    = 0.5 * (w2 - 1)
    r2y    = 0.5 * (h2 - 1)
    
    ## search range
    class SearchRange():
        def __init__(self, minScale, maxScale,\
                     minRotation, maxRotation, \
                     minTx=0, maxTx=0, minTy=0, maxTy=0):
            self.minScale       = minScale
            self.maxScale       = maxScale
            self.minRotation    = minRotation
            self.maxRotation    = maxRotation
            self.minTx          = minTx
            self.maxTx          = maxTx
            self.minTy          = minTy
            self.maxTy          = maxTy
    searchRange = SearchRange(
        minScale    = 0.5,
        maxScale    = 2,
        minRotation = -np.pi,
        maxRotation = np.pi,
    )
    searchRange.minTx       = -(r2x - r1x * searchRange.minScale)
    searchRange.maxTx       = r2x - r1x * searchRange.minScale
    searchRange.minTy       = -(r2y - r1y * searchRange.minScale)
    searchRange.maxTy       = r2y - r1y * searchRange.minScale


    # check
    assert (searchRange.minScale    >=0)        & (searchRange.minScale     <=1)
    assert (searchRange.maxScale    >=1)        & (searchRange.maxScale     <=5)
    assert (searchRange.minRotation >=-np.pi)   & (searchRange.minRotation  <=0)
    assert (searchRange.maxRotation >=0)        & (searchRange.maxRotation  <= np.pi)

    # copy params
    minScale    = searchRange.minScale
    maxScale    = searchRange.maxScale
    minRotation = searchRange.minRotation
    maxRotation = searchRange.maxRotation
    minTx       = max(searchRange.minTx,-(r2x-r1x*minScale))
    maxTx       = min(searchRange.maxTx,r2x-r1x*minScale)
    minTy       = max(searchRange.minTy,-(r2y-r1y*minScale))
    maxTy       = min(searchRange.maxTy,r2y-r1y*minScale)
    
    ## parametrize the initial grid
    bounds, steps = GenerateGrid(w1,h1,delta,minTx,maxTx,minTy,maxTy,minRotation,maxRotation,minScale,maxScale)
    
    bestConfig,bestTransMat,sampledError = \
        FindBestTransformation(template,img,bounds,steps,epsilon,delta,photometricInvariance, templateMask)

    return bestConfig,bestTransMat,sampledError

def MakeOdd(img):
    if len(img.shape) == 2:
        h, w = img.shape[:2]
    elif len(img.shape) == 3:
        h, w, d = img.shape
        if d >> 1 << 1== d:
            img = img[:,:,:-1]

    croppedH = 0
    croppedW = 0

    if h >> 1 << 1 == h:
        cropppedH = 1
        img = img[:-1]
    if w >> 1 << 1 == w:
        croppedW = 1
        img = img[:, :-1]
    return img

def GenerateGrid(sourceW, sourceH, delta,minTx,maxTx,minTy,maxTy,minR,maxR,minS,maxS):
    class GridValue():
        def __init__(self, tx, ty, r, s):
            self.tx = tx
            self.ty = ty
            self.r  = r
            self.s  = s
        
    bounds = GridValue(
        tx = [minTx, maxTx],
        ty = [minTy, maxTy],
        r  = [minR, maxR],
        s  = [minS, maxS]
    )
    
    steps = GridValue(
        tx = delta * sourceW / np.sqrt(2),
        ty = delta * sourceH / np.sqrt(2),
        r  = delta * np.sqrt(2),
        s  = delta / np.sqrt(2)
    )
    
    return bounds, steps

def FindBestTransformation(I1, I2, bounds, steps, epsilon, delta, photometricInvariance, templateMask):
    ## %% blur in main loop - this reduces the total-variation and gives better results
    assert len(I1.shape) == 2, 'the img must be gray'
    origI1 = I1.copy()
    origI2 = I2.copy()
    
    h1, w1 = I1.shape
    
    ## generate Theta(1/eps^2) random points
    numPoints = round(10 / epsilon ** 2)
    xs, ys = getPixelSample(templateMask, numPoints)
    
    ## generate the Net
    configs, gridSize = CreateListOfConfigs(bounds, steps)
    
    if configs.shape[0] > 71000000:
        raise 'more than 35 million configs!'
        
    # main loop
    deltaFact = 1.511
    level = 0
    bestDists = []
    perRoundNumConfigs = []
    perRoundNumGoodConfigs = []
    perRoundOrig_percentage = []
    bestGridVec = []
    newDelta = delta

    while 1:
        level = level + 1
        
        # if (isGrayscale)  the pic is must gray, bcz we use the assert
        blur_sigma = 1.5 + 0.5 / deltaFact**(level-1) 
        blur_size = np.ceil(4 * blur_sigma).astype(np.int16)
        if blur_size % 2 == 0:
            blur_size += 1
        
        I1 = cv2.GaussianBlur(origI1, (blur_size, blur_size), blur_sigma, cv2.BORDER_REFLECT)
        I2 = cv2.GaussianBlur(origI2, (blur_size, blur_size), blur_sigma, cv2.BORDER_REFLECT)
        
        h2, w2 = I2.shape
        
        r1x = 0.5*(w1-1)
        r1y = 0.5*(h1-1)
        r2x = 0.5*(w2-1)
        r2y = 0.5*(h2-1)

        # 0] if limited rotation range -fileter out illegal rotations
        if bounds.r[0] > -np.pi or bounds.r[1] > np.pi:
            minRot = bounds.r[0]
            maxRot = bounds.r[1]
            # totla rotations in the range [0, 2*pi]
            totalRots = divmod(configs[:,2] + configs[:,5], 2*np.pi)[1]
            # total rotation in the range [-pi, pi]
            # TODebug:
            totalRots[totalRots > np.pi] = totalRots[totalRots > np.pi] - 2*np.pi
            # filtering
            configs = configs[np.logical_and(totalRots > minRot , totalRots < maxRot), :]
        
        # 1] translate config vectors to matrix form
        # Configs2AffineMEX = tic
        print('---- Configs2Affine, with %d configs -----\n'%configs.shape[0])
        
        matrixConfigs_mex, insiders = \
            Configs2Affine_mex(configs, h1, w1, h2, w2, r1x, r1y, r2x, r2y)
        
        inBoundaryInds = np.nonzero(insiders)[0]
        matrixConfigs_mex = matrixConfigs_mex[inBoundaryInds, :]
        origNumConfigs = configs.shape[0]
        
        configs = configs[inBoundaryInds, :]
        
        # 2] evaluate all configurations
        EvaluateConfigsMEX = time.time()

        distances = EvaluateConfigs_mex(I1, I2, matrixConfigs_mex, xs, ys, photometricInvariance)
        print('---- Evaluate Configs vectorized, with %d configs ----\n' % configs.shape[0])
        
        EvaluateConfigs_mex_time = time.time() - EvaluateConfigsMEX
        print(f'@@@@ Time cost: {EvaluateConfigs_mex_time:.3f} sec.')

        bestDist, ind = np.min(distances), np.argmin(distances)
        bestConfig = configs[ind, :]
        bestTransMat = CreateAffineTransformation(configs[ind, :])
        
        # 3] choose the 'surviving' configs and delta for next round
        goodConfigs, tooHighPercentage, extremHighPrecent, veryLowPercentage, orig_percentage, thresh = \
            GetGoodConfigsByDistance(configs, bestDist, newDelta, distances, bestGridVec)
            
        numGoodConfigs = goodConfigs.shape[0]
        print('$$$ best Dist = %.3f\n' % bestDist)
        print('$$ numGoodConfigs: %d (out of %d), orig precentage: %.4f, bestDist: %.4f, thresh: %.4f\n' %
              (goodConfigs.shape[0], configs.shape[0], orig_percentage, bestDist, thresh))
        
        # collect round stats
        if len(bestDists) + 1 == level:
            bestDists.append(bestDist)
            perRoundNumConfigs.append(origNumConfigs)
            perRoundNumGoodConfigs.append(numGoodConfigs)
            perRoundOrig_percentage.append(orig_percentage)
        else:
            bestDists[level - 1] = bestDist
            perRoundNumConfigs[level - 1] = origNumConfigs
            perRoundNumGoodConfigs[level - 1] = numGoodConfigs
            perRoundOrig_percentage[level - 1] = orig_percentage
        
        # 4] break conditions of Branch-and-Bound
        conditions = [0] * 6
        conditions[0] = bestDist < 0.005 # % good enough 1
        conditions[1] = (level > 5) and (bestDist < 0.01) #  % good enough 2
        conditions[2] = (level >= 20) #  % enough levels
        conditions[3] = ((level > 3) and (bestDist > np.mean(bestDists[level-3:level-1])*0.97)) # % no improvement in last 3 rounds
        conditions[4] = ((level > 2) and (numGoodConfigs>1000) and extremHighPrecent ) # % too high expansion rate
        conditions[5] = ((level > 3) and (numGoodConfigs>1000) and (numGoodConfigs>50*min(perRoundNumGoodConfigs))) # % a deterioration in the focus

        if any(conditions):
            print(f'breaking BnB at level {level} due to conditions: ', np.nonzero(conditions)[0])
            print('best distances by round: ', bestDists)
            print('num configs per round(K):  ', [ i/1000 for i in perRoundNumConfigs])
            print('# good configs per round: ', perRoundNumGoodConfigs)
            print('percentage to expand:   ', perRoundOrig_percentage)
            break

        # 6] debug: visualize on histogram
        
        # 7] expand 'surviving' configs for next round
        #    # ('restart' = [with samller delta] if not too many configs and not too high percentage of configs to expand)
        if (not veryLowPercentage) and \
            ( (tooHighPercentage and (bestDist > 0.1) and ((level == 1) and (origNumConfigs < 7.5 * 1e+6)) ) or \
              (                      (bestDist > 0.15) and ((level == 1) and (origNumConfigs < 5 * 1e+6))  )    ) :
            fact = 0.9 
            print(' ##### RESRARTING!!! changing from delta: %.3f, to delta: %.3f\n' %(newDelta, newDelta*fact))
            newDelta = newDelta * fact
            level = 0
            steps.tx = fact * steps.tx
            steps.ty = fact*steps.ty
            steps.r = fact*steps.r
            steps.s = fact*steps.s
            configs,gridSize = CreateListOfConfigs(bounds,steps)
        else:
            prevDelta = newDelta
            newDelta = newDelta / deltaFact
            print('##### CONTINUING!!! prevDelta = %.3f,  newDelta = %.3f \n'%(prevDelta,newDelta))
            
            # expand the good configs
            expandType = 'randomExpansion' # 'fullExpansion' 
            if expandType == 'randomExpansion':
                expandedConfigs = ExpandConfigsRandom(goodConfigs,steps,level,80,deltaFact)
            configs = np.vstack([goodConfigs, expandedConfigs])
        
        print('***')
        print('*** level %d:|goodConfigs| = %d, |expandedConfigs| = %d'%(level, numGoodConfigs, configs.shape[0]))
        print('***')
        
        # 8] refresh random points
        xs, ys = getPixelSample(templateMask, numPoints)
        
    ## debug error
    ## for output
    sampledError = bestDist
        
    return bestConfig, bestTransMat, sampledError

def getPixelSample(mask, numPoints):
    h, w = mask.shape
    nzeroInd = np.nonzero(mask.reshape(-1))[0]
    rng = np.random.default_rng()
    idx = rng.choice(nzeroInd, numPoints, replace=False)
    ys, xs = np.divmod(idx, w)
    return xs, ys

def CreateListOfConfigs(bounds, steps): 
    # return configs' shape is 42525x6
    tx_steps = np.arange(bounds.tx[0], bounds.tx[1] + 0.5*steps.tx, steps.tx)
    ty_steps = np.arange(bounds.ty[0], bounds.ty[1] + 0.5*steps.ty, steps.ty)
    r_steps  = np.arange(-np.pi, np.pi, steps.r)
    s_steps  = np.arange(bounds.s[0], bounds.s[1] + 0.5*steps.s, steps.s)
                        
    # number of steps
    ntx_steps = len(tx_steps)
    nty_steps = len(ty_steps)
    ns_steps = len(s_steps)
    nr_steps = len(r_steps)
    
    # second rotation is s special case (can be limited to a single quartile)
    #TODO: to understand
    quartile1_r_steps = r_steps[r_steps < -np.pi/2 + steps.r/2]
    NR2_steps = len(quartile1_r_steps)
    
    # gridsize
    gridSize = ntx_steps * nty_steps * (ns_steps**2) * (nr_steps * NR2_steps)
    
    configs = createlist_mex(ntx_steps,nty_steps,nr_steps,NR2_steps,ns_steps,tx_steps,ty_steps,r_steps,s_steps,gridSize)
    return configs, gridSize

def createlist_mex(ntx_steps, nty_steps, nr_steps, NR2_steps, ns_steps, tx_steps, ty_steps, r_steps, s_steps, gridSize):
    configs = np.zeros([gridSize, 6])

    gridInd = 0
    for tx_ind in range(ntx_steps):
        tx = tx_steps[tx_ind]
        for ty_ind in range(nty_steps):
            ty = ty_steps[ty_ind]
            for r1_ind in range(nr_steps):
                r1 = r_steps[r1_ind]
                for r2_ind in range(NR2_steps):
                    r2 = r_steps[r2_ind]
                    for sx_ind in range(ns_steps):
                        sx = s_steps[sx_ind]
                        for sy_ind in range(ns_steps):
                            sy = s_steps[sy_ind]
                            configs[gridInd, 0] = tx
                            configs[gridInd, 1] = ty
                            configs[gridInd, 2] = r2
                            configs[gridInd, 3] = sx
                            configs[gridInd, 4] = sy
                            configs[gridInd, 5] = r1
                            gridInd += 1
    return configs

def Configs2Affine_mex(configs, sourceH, sourceW, targetH, targetW, r1x, r1y, r2x, r2y):
    numConfigs = configs.shape[0]
    affines = np.zeros([numConfigs, 6], dtype=np.float64)
    insiders = np.zeros([numConfigs,], dtype=np.int8)


    affines[:, 0]  = configs[:, 3]*np.cos(configs[:, 5])*np.cos(configs[:, 2]) - configs[:, 4]*np.sin(configs[:, 5])*np.sin(configs[:, 2])
    affines[:, 1]  = - configs[:, 3]*np.cos(configs[:, 5])*np.sin(configs[:, 2]) - configs[:, 4]*np.cos(configs[:, 2])*np.sin(configs[:, 5])
    affines[:, 2]  = configs[:, 0]
    affines[:, 3]  = configs[:, 3]*np.cos(configs[:, 2])*np.sin(configs[:, 5]) + configs[:, 4]*np.cos(configs[:, 5])*np.sin(configs[:, 2])
    affines[:, 4]  = configs[:, 4]*np.cos(configs[:, 5])*np.cos(configs[:, 2]) - configs[:, 3]*np.sin(configs[:, 5])*np.sin(configs[:, 2])
    affines[:, 5]  = configs[:, 1]

    c1x = affines[:, 0]  *(1-(r1x+1)) + affines[:, 1]*(1-(r1y+1)) + (r2x+1) + configs[:, 0]
    c1y = affines[:, 3]*(1-(r1x+1)) + affines[:, 4]*(1-(r1y+1)) + (r2y+1) + configs[:, 1]
    c2x = affines[:, 0]  *(sourceW-(r1x+1)) + affines[:, 1]*(1-(r1y+1)) + (r2x+1) + configs[:, 0]
    c2y = affines[:, 3]*(sourceW-(r1x+1)) + affines[:, 4]*(1-(r1y+1)) + (r2y+1) + configs[:, 1]
    c3x = affines[:, 0]  *(sourceW-(r1x+1)) + affines[:, 1]*(sourceH-(r1y+1)) + (r2x+1) + configs[:, 0]
    c3y = affines[:, 3]*(sourceW-(r1x+1)) + affines[:, 4]*(sourceH-(r1y+1)) + (r2y+1) + configs[:, 1]
    c4x = affines[:, 0]  *(1-(r1x+1)) + affines[:, 1]*(sourceH-(r1y+1)) + (r2x+1) + configs[:, 0]
    c4y = affines[:, 3]*(1-(r1x+1)) + affines[:, 4]*(sourceH-(r1y+1)) + (r2y+1) + configs[:, 1]

    flag = np.array([c1x>-10, c1x<targetW+10, c1y>-10, c1y<targetH+10,  
            c2x>-10, c2x<targetW+10, c2y>-10, c2y<targetH+10,  
            c3x>-10, c3x<targetW+10, c3y>-10, c3y<targetH+10,  
            c4x>-10, c4x<targetW+10, c4y>-10, c4y<targetH+10 ])
    insiders = np.sum(flag, axis=0)
    insiders = np.where(insiders == 16, 1, 0)

    # for i in range(numConfigs):
    #     tx = configs[i, 0]
    #     ty = configs[i, 1]
    #     r2 = configs[i, 2]
    #     sx = configs[i, 3]
    #     sy = configs[i, 4]
    #     r1 = configs[i, 5]

    #     affines[i, 0]  = sx*np.cos(r1)*np.cos(r2) - sy*np.sin(r1)*np.sin(r2)
    #     affines[i, 1]  = - sx*np.cos(r1)*np.sin(r2) - sy*np.cos(r2)*np.sin(r1)
    #     affines[i, 2]  = tx
    #     affines[i, 3]  = sx*np.cos(r2)*np.sin(r1) + sy*np.cos(r1)*np.sin(r2)
    #     affines[i, 4]  = sy*np.cos(r1)*np.cos(r2) - sx*np.sin(r1)*np.sin(r2)
    #     affines[i, 5]  = ty

    #     c1x = affines[i, 0]  *(1-(r1x+1)) + affines[i, 1]*(1-(r1y+1)) + (r2x+1) + tx
    #     c1y = affines[i, 3]*(1-(r1x+1)) + affines[i, 4]*(1-(r1y+1)) + (r2y+1) + ty
    #     c2x = affines[i, 0]  *(sourceW-(r1x+1)) + affines[i, 1]*(1-(r1y+1)) + (r2x+1) + tx
    #     c2y = affines[i, 3]*(sourceW-(r1x+1)) + affines[i, 4]*(1-(r1y+1)) + (r2y+1) + ty
    #     c3x = affines[i, 0]  *(sourceW-(r1x+1)) + affines[i, 1]*(sourceH-(r1y+1)) + (r2x+1) + tx
    #     c3y = affines[i, 3]*(sourceW-(r1x+1)) + affines[i, 4]*(sourceH-(r1y+1)) + (r2y+1) + ty
    #     c4x = affines[i, 0]  *(1-(r1x+1)) + affines[i, 1]*(sourceH-(r1y+1)) + (r2x+1) + tx
    #     c4y = affines[i, 3]*(1-(r1x+1)) + affines[i, 4]*(sourceH-(r1y+1)) + (r2y+1) + ty

    #     flag = [c1x>-10, c1x<targetW+10, c1y>-10, c1y<targetH+10,  
    #             c2x>-10, c2x<targetW+10, c2y>-10, c2y<targetH+10,  
    #             c3x>-10, c3x<targetW+10, c3y>-10, c3y<targetH+10,  
    #             c4x>-10, c4x<targetW+10, c4y>-10, c4y<targetH+10 ]
    #     insiders[i] = 1 if all(flag) else 0
    return affines, insiders

def EvaluateConfigs_mex(I1, I2, configs, xs, ys, photometricInvariance):
    xs = xs.astype(np.int32)
    ys = ys.astype(np.int32)
    h1, w1 = I1.shape[:2]
    h2, w2 = I2.shape[:2]
    
    # I1 = np.arange(h1*w1) + 1
    # I2 = np.arange(h2*w2) + 1
    img1 = I1.reshape(-1,)
    img2 = np.zeros([5 * h2 * w2, ])
    img2[2 * h2 * w2 : 3 * h2 * w2] = I2.reshape(-1,)

    numConfigs = configs.shape[0]
    numPoints  = len(xs)

    r1x = 0.5*(w1-1)
    r1y = 0.5*(h1-1)
    r2x = 0.5*(w2-1)
    r2y = 0.5*(h2-1)

    distances   = np.zeros([numConfigs, ])
    xs_centered = np.zeros([1, numPoints], dtype=np.int64)
    ys_centered = np.zeros([1, numPoints], dtype=np.int64)
    valsI1      = np.zeros([numPoints, 1], dtype=np.float64)

    # /*Centered pointes*/   ?????  what is center point
    xs_centered = xs - (r1x + 1)
    ys_centered = ys - (r1y + 1)
    # /*Precalculating source point indices into I1 (and the values themselves)*/
    valsI1 = img1[ys*w1 + xs]

    tmp1 = (r2x+1) + configs[:, 2] + 0.5
    tmp2 = (r2y+1) + configs[:, 5] + 0.5 + 2*h2

    targetPoint_x = np.dot(configs[:, 0].reshape(-1,1), xs_centered.reshape(1, -1)) + \
        np.dot(configs[:, 1].reshape(-1,1), ys_centered.reshape(1,-1)) + tmp1.reshape(-1,1)
    targetPoint_y = np.dot(configs[:, 3].reshape(-1,1), xs_centered.reshape(1, -1)) + \
        np.dot(configs[:, 4].reshape(-1,1), ys_centered.reshape(1,-1)) + tmp2.reshape(-1,1)    # includes rounding
    targetPoint_x = targetPoint_x.astype(np.int64)
    targetPoint_y = targetPoint_y.astype(np.int64)
    targetInd     = targetPoint_y  * w2 + targetPoint_x

    distances = np.mean(np.fabs(valsI1.reshape(1,-1) - img2[targetInd]), axis=1)

    # # // MAIN LOOP
    # for i in tqdm(range(numConfigs)):
    #     a11 = configs[i, 0]
    #     a12 = configs[i, 1]
    #     a13 = configs[i, 2]
    #     a21 = configs[i, 3]
    #     a22 = configs[i, 4]
    #     a23 = configs[i, 5]

    #     score = 0
    #     tmp1 = (r2x+1) + a13 + 0.5
    #     tmp2 = (r2y+1) + a23 + 0.5 + 2*h2

    #     if not photometricInvariance:
    #         targetPoint_x = (a11 * xs_centered + a12 * ys_centered + tmp1).astype(np.int64)
    #         targetPoint_y = (a21 * xs_centered + a22 * ys_centered + tmp2).astype(np.int64)    # includes rounding
    #         targetInd     = targetPoint_y  * w2 + targetPoint_x        # -1 is for c
    #         score = np.sum(np.fabs(valsI1 - img2[targetInd]))
    #     else:
    #         pass
    #     distances[i] = score/numPoints
    return distances

def CreateAffineTransformation(config):
    tx, ty, r2, sx, sy, r1 = config[:]
    R1 = np.array([[np.cos(r1), -np.sin(r1)], [np.sin(r1), np.cos(r1)]])
    R2 = np.array([[np.cos(r2), -np.sin(r2)], [np.sin(r2), np.cos(r2)]])
    S  = np.array([[sx, 0], [0, sy]])
    A  = np.array([[0, 0, tx], [0, 0, ty], [0, 0, 1]])
    A[0:2, 0:2] = np.dot(np.dot(R1, S), R2)
    return A
    
def GetGoodConfigsByDistance(configs, bestDist, newDelta, distances, bestGridVec):
    thresh = bestDist + GetThreshPerDelta(newDelta)
    goodConfigs = configs[distances < thresh, :]
    numGoodConfigs = goodConfigs.shape[0]
    orig_percentage = numGoodConfigs / configs.shape[0]
    
    # too  many good configs - reducing threshold
    while numGoodConfigs > 27000:
        thresh = thresh * 0.99
        goodConfigs = configs[distances <= thresh, :]
        numGoodConfigs = goodConfigs.shape[0]
    if len(goodConfigs) == 0:
        thresh = min(distances)
        goodConfigs = configs[distances <= thresh, :]
        if goodConfigs.shape[0] > 10000:
            inds = np.nonzero(distances <= thresh)[0]
            goodConfigs = configs[inds[:100], :]
    
    tooHighPercentage = orig_percentage > 0.05
    veryLowPercentage = orig_percentage < 0.01
    extremelyHighPercentage = orig_percentage > 0.2
    
    if len(bestGridVec) != 0:
        exists, bestGridInd = IsMemberApprox(goodConfigs, bestGridVec, 1e-6)
        if exists is None:
            print('problem with configs')
    return goodConfigs, tooHighPercentage, extremelyHighPercentage, veryLowPercentage, orig_percentage, thresh
            
def GetThreshPerDelta(delta):
    p = [0.1341, 0.0278]
    safety = 0.02
    thresh = p[0] * delta + p[1] - safety
    return thresh

def IsMemberApprox(A, row, err):
    res = 0
    for i in range(A.shape[0]):
        if np.linalg.norm(A[i, :] - row) < err:
            res = 1
            return res, i
    return None, -1

def ExpandConfigsRandom(configs, steps, level, npoints, deltaFact):
    fact = deltaFact ** level
    halfstep_tx = steps.tx / fact
    halfstep_ty = steps.ty / fact
    halfstep_r = steps.r / fact
    halfstep_s = steps.s / fact
    
    numConfigs = configs.shape[0]
    randvec = np.floor(3 * np.random.uniform(0, 1, size=(npoints*numConfigs, 6))- 1)
    expanded = np.repeat(configs, npoints, axis=0)
    ranges = np.array([[halfstep_tx, halfstep_ty, halfstep_r, halfstep_s, halfstep_s, halfstep_r]])
    expandedConfigs = expanded + randvec * np.repeat(ranges, npoints*numConfigs, axis=0)
    return expandedConfigs

def MatchingResult(I1,I2,a):
    templateMask = np.ones(I1.shape)
    I1[templateMask == 0] = 0
    tx = a[0,2]
    ty = a[1,2]
    h1, w1 = I1.shape
    h2, w2 = I2.shape

    r1x = 0.5*(w1-1)
    r1y = 0.5*(h1-1)
    r2x = 0.5*(w2-1)
    r2y = 0.5*(h2-1)

    [ys, xs] = np.nonzero(templateMask)

    a2x2= a[:2, :2]
    cornersX = np.array([1, w1, w1, 1])
    cornersY = np.array([1, 1, h1, h1])

    t = np.vstack([cornersX - (r1x+1), cornersY - (r1y+1)])
    cornerA = np.dot(a2x2, t)
    cornerAxs = np.round(cornerA[0, :] + r2x + 1 + a[0,2])
    cornerAys = np.round(cornerA[1, :] + r2y + 1 + a[1,2])
    cornerAxs = cornerAxs.tolist() + [cornerAxs[0]]
    cornerAys = cornerAys.tolist() + [cornerAys[0]]

    plt.imshow(I2)
    plt.plot(cornerAxs, cornerAys,'-r')
    plt.plot(cornerAxs, cornerAys,'*r')
    plt.show()

if __name__ == '__main__':
    # img = cv2.imread(r'imgPairs\image__1.png', 0)
    # template = cv2.imread(r'imgPairs\image_for_template__1.png', 0)
    img = cv2.imread(r'example\image.png', 0)
    template = cv2.imread(r'example\template.png', 0)    
    img = img / 255
    template = template / 255
    # FastMatch(templage, img)
    start_time = time.time()
    bestConfig, bestTransMat, sampledError = FastMatch(template, img)
    print('Time cost: %.2f'%(time.time() - start_time))
    # a = np.array([[0.5160,     0.6628,   46.4694],
    #               [-1.5696,   -0.2140,   -2.3668],
    #               [0,               0,    1.0000]])
    MatchingResult(template, img, bestTransMat)