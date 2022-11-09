import os
import torch
import torch.optim
import torch.nn as nn

from tqdm import tqdm
from data import generate_loader
from option import get_option
from model import YOLOv3
from loss import YoloLoss
from utils.bbox_tools import get_evaluation_bboxes
from utils.mAP import mean_average_precision


class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.model = YOLOv3(num_classes=opt.num_classes).to(self.dev)
        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
            self.model.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.device_ids).to(self.dev)

        print("# params:", sum(map(lambda x: x.numel(), self.model.parameters())))

        self.scaled_anchors = (torch.tensor(opt.Anchors) * torch.tensor(opt.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(self.dev)
        self.loss_fn = YoloLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), opt.lr, weight_decay=opt.weight_decay)

        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.val_loader = generate_loader('val', opt)
        print("validation set ready")
        self.best_mAP, self.best_epoch = 0, 0

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            self.model.train()
            loop = tqdm(self.train_loader, leave=True)
            losses = []

            for _, (img, output_label) in enumerate(loop):
                img = img.to(self.dev)
                y0, y1, y2 = (
                    output_label[0].to(self.dev),
                    output_label[1].to(self.dev),
                    output_label[2].to(self.dev)
                )

                preds = self.model(img)
                loss = (
                    self.loss_fn([preds[0], y0, self.scaled_anchors[0]])
                    + self.loss_fn([preds[1], y1, self.scaled_anchors[1]])
                    + self.loss_fn([preds[2], y2, self.scaled_anchors[2]])
                )

                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # update progress bar
                mean_loss = sum(losses) / len(losses)
                loop.set_postfix(loss=mean_loss)

            # evaluation
            self.model.eval()
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.val_loader, 
                self.model, 
                iou_threshold=opt.nms_iou_threshold,
                anchors=opt.Anchors,
                threshold=opt.conf_threshold,
                device=self.dev
                )

            mAP = mean_average_precision(
                pred_boxes, 
                true_boxes,
                iou_threshold= opt.map_iou_threshold,
                num_classes=opt.num_classes
                )

            if mAP >= self.best_mAP:
                self.best_mAP, self.best_epoch = mAP, epoch
                self.save()

            print("Epoch [{}/{}] Loss: {:.3f}, Val mAP: {:.3f}".format(epoch+1, opt.n_epoch, mean_loss, mAP))
            print("Best mAP & epch : {:.2f} @ {}".format(self.best_mAP, self.best_epoch+1))


    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
        torch.save(self.model.state_dict(), save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()

if __name__ == "__main__":
    main()