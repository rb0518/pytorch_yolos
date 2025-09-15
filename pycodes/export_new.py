import argparse
import torch
import cv2
import torchvision

def parse_opt(know=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', type=str, default='weights/yolov5s.pt', help='./run/project name/ must be use torch.save(model, pt)')
    parser.add_argument('--export_file', type=str, default='weights/yolov5s.script.pt',         help='program store folder: ./run/project')
    parser.add_argument('--test_file',   type=str, default='data/images/bus.jpg',        help='path/to/test image')
    return parser.parse_args()


def main(opt):
    model_ = torch.load(opt.weight_file, weights_only=False, map_location=torch.device('cuda'))['model'].float()
    model_.to('cuda')
    model_.eval()
    test = torch.zeros(1, 3, 640, 640).float()
    test = test.to(torch.device('cuda'))
    trace_module = torch.jit.trace(model_, test)
    torch.jit.save(trace_module, opt.export_file)

if __name__ == "__main__":
    opt = parse_opt()

    main(opt)