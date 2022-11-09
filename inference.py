import os
import torch
from model import YOLOv3
from data import generate_loader
from option import get_option
from utils.bbox_tools import plot_image, cells_to_bboxes
from utils.nms import non_max_suppression

def inference(opt):
    dev = dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

    data_loader = generate_loader('test', opt)
    print("data load complete")

    model = YOLOv3(opt).to(dev)
    load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
    model.load_state_dict(torch.load(load_path))
    print("model construct complete")

    for img, _ in data_loader:
        img = img.to(dev)
    
        for idx in range(opt.eval_batch_size):
            bboxes = cells_to_bboxes(model(img), opt.Anchors, opt.S)
            nms_bboxes = non_max_suppression(
                bboxes[idx], 
                iou_threshold=opt.nms_iou_threshold, 
                class_threshold=opt.conf_threshold
            )
            plot_image(img[idx].to("cpu"), nms_bboxes, opt)


if __name__ =='__main__':
    opt = get_option()
    torch.manual_seed(opt.seed)
    inference(opt)