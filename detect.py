import argparse
import shutil
import time
from pathlib import Path
from sys import platform
import moviepy.editor as mpe
import numpy as np
import os

from models import *
from utils.datasets import *
from utils.utils import *
import logging


def detect(
        cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.45,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]
    
    time_frame = []
    for i, (path, img, im0) in enumerate(dataloader):
        t = time.time()
        if webcam:
            print('webcam frame %g: ' % (i + 1), end='')
        else:
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            try :
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]
                #print(detections)

                # Rescale boxes from 416 to true image size
                detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape)
                #print(detections[:, :4])

                # Print results to screen
                unique_classes = detections[:, -1].cpu().unique()
                for c in unique_classes:
                    n = (detections[:, -1].cpu() == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write('%g %g %g %g %g %g\n' %
                                    (x1, y1, x2, y2, cls, cls_conf * conf))

                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])
            except:
                print("sth wrong")

        dt = time.time() - t
        time_frame.append(dt)
        print('Done. (%.3fs)' % dt)

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)

        if webcam:  # Show live webcam
            #cv2.imshow(weights + ' - %.2f FPS' % (1 / dt), im0)
            cv2.imshow("im",im0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows() 

    if save_images and (platform == 'darwin'):  # linux/macos
        os.system('open ' + output + ' ' + save_path)
    
    print('The Processing time for frame is (%.3fs)' % float(sum(time_frame)))
    
    return time_frame


if __name__ == '__main__':
    logging.basicConfig(filename='log.log',level=logging.DEBUG)
    start_time = time.time()
    t_preprocess = time.time()
    video = mpe.VideoFileClip('test2ot.mp4')
    #np_frame = video.get_frame(2) # get the frame at t=2 seconds
    c=0
    t = np.arange(0,32,0.03)

    if os.path.exists('test2ot'):
        shutil.rmtree('test2ot')  # delete output folder
    os.makedirs('test2ot')

    for i in t:
        video.save_frame('./test2ot/'+str(i)+'.jpg', t=i) # save frame at t=2 as JPEG
    
    dt_preprocess = time.time() - t_preprocess

    print('The preprocess time is (%.3fs)'% dt_preprocess)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)




    

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
    
    t_back2video = time.time()
    import moviepy.editor as mpe
    pic_list = os.listdir('./output')
    pic_list.sort(key=lambda x: float(x.split('.j')[0]))
    pic_lis = []
    for i in pic_list:
        pic_lis.append('./output/'+i)
    clip = mpe.ImageSequenceClip(pic_lis, fps=30)
    clip.write_videofile('./output_video/video_test.mp4', fps=30)

    dt_back2video = time.time() - t_back2video
    print('The preprocess time is (%.3fs)'% dt_back2video)
    #total_time = dt_back2video+dt_preprocess+float(sum(time_frame)
    #print(total_time)
    #print(time_frame)
    logging.info('--- %s seconds ---' % (time.time()-start_time))


