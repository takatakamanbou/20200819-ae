from IPython.display import display, Image
import numpy as np
import cv2

### 画像表示用の関数
#
def displayImage(img):
    
    if type(img) is np.ndarray:               # 渡されたのが np.ndarray だった（OpenCVの画像はこの形式)
        rv, buf = cv2.imencode('.png', img)  # PNG形式のバイト列に変換
        if rv:
            display(Image(data = buf.tobytes()))   # 変換できたらバイト列を渡して表示
            return
    elif type(img) is str:                         # 渡されたのが文字列だった
        display(Image(filename = img))
        return
    
    print('displayImage: error')


#####  データの最初の nx x ny 枚を可視化
#
def mosaicImage(dat, nx, ny, nrow = 128, ncol = 96, gap = 4):

    # 並べた画像の幅と高さ
    width  = nx * (ncol + gap) + gap
    height = ny * (nrow + gap) + gap

    # 画像の作成
    img = np.zeros((height, width, 3), dtype = int) + 128
    for iy in range(ny):
        lty = iy*(nrow + gap) + gap
        for ix in range(nx):
            if iy*nx+ix >= dat.shape[0]:
                break
            ltx = ix*(ncol + gap) + gap
            img[lty:lty+nrow, ltx:ltx+ncol, :] = dat[iy*nx+ix, :].reshape((nrow, ncol, 3))
            
    return img