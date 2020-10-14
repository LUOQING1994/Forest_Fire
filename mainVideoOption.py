import cv2
import testModel
import torch
from flask import Flask
app = Flask(__name__)

@app.route('/startModel')
def startModel():
    # 调用模型配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    # 视频输入配置
    url = './video/video.avi'
    videos = cv2.VideoCapture(url)
    rval, flag_ = videos.read()
    # 视频输出配置
    video_name = 'output.avi'
    frames = videos.get(cv2.CAP_PROP_FPS)
    # img_size = (cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)
    img_size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(video_name, fourcc, frames, img_size)

    while rval:
        rval, flag_ = videos.read()
        if not rval:
            break
        # 调用测试模型
        result_image = testModel.callNetworkModel(flag_, device)
        resize_img = cv2.resize(result_image, img_size)
        # 转换成uint8 防止显示时正常 保存时全黑的情况
        noise_img_norm = cv2.normalize(resize_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("flag_", noise_img_norm)
        # cv2.waitKey(-1)
        # 制作视频
        video_writer.write(noise_img_norm)
        cv2.waitKey(100)
    video_writer.release()
    videos.release()
    cv2.destroyAllWindows()
    return '视频分析完毕！'

if __name__ == "__main__":
    app.run()
