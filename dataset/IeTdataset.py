import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset
from pathlib import Path 
import bisect
import numpy as np
import cv2
from dataset.label import CLASSES
from dataset.directory import DSECDirectory
from dataset.io import extract_from_h5_by_timewindow
from dataset.visualize import render_object_detections_on_image, render_events_on_image
from dataset.preprocessing import compute_img_idx_to_track_idx
import util.transforms as T
from PIL import Image
import random
from SODFormer.util.misc import NestedTensor, nested_tensor_from_tensor_list

# def make_transforms(image_set='train'):
#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
#                     [0.927, 0.856, 0.899], [0.147, 0.223, 0.183])
#     ])

#     scales = [320, 352, 384, 416, 448, 480]
#     if image_set == 'train':
#         return T.Compose([
#             T.RandomResize(scales, max_size=640),
#             normalize,
#         ])

#     if image_set == 'val' or image_set == 'test':
#         return T.Compose([
#             T.RandomResize([480], max_size=640),
#             normalize,
#         ])

#     raise ValueError(f'unknown {image_set}')

def make_transforms(image_set='train'):
    # 只保留 ToTensor
    return T.Compose([
        T.ToTensor(),
    ])

class DSECDet:
    def __init__(self, root: Path, batch_size, split: str="train", sync: str="front", debug: bool=False, split_config=None,
                 transform=None):
        """
        root: Root to the the DSEC dataset (the one that contains 'train' and 'test'
        split: Can be one of ['train', 'test']
        window_size: Number of microseconds of data
        sync: Can be either 'front' (last event ts), or 'back' (first event ts). Whether the front of the window or
              the back of the window is synced with the images.

        Each sample of this dataset loads one image, events, and labels at a timestamp. The behavior is different for 
        sync='front' and sync='back', and these are visualized below.

        Legend: 
        . = events
        | = image
        L = label

        sync='front'
        -------> time
        .......|
               L

        sync='back'
        -------> time
        |.......
               L
        
        """
        assert root.exists()
        assert split in ['train', 'test', 'val']
        assert (root / split).exists()
        assert sync in ['front', 'back']

        self.debug = debug
        self.classes = CLASSES

        if split == 'train':
            self.root = root / "train"
        elif split == 'val':
            self.root = root / "val"
        elif split == 'test':
            self.root = root / "test"

        self.sync = sync

        self.height = 480
        self.width = 640

        self.prepare = Prepare('dvs')
        self.transform = transform

        self.directories = dict()
        self.img_idx_track_idxs = dict()

        #获取目录下的所有文件夹名称
        if split_config is None:
            self.subsequence_directories = list(self.root.glob("*/"))
        else:
            available_dirs = list(self.root.glob("*/"))
            self.subsequence_directories = [self.root / s for s in split_config[split] if self.root / s in available_dirs]
        
        #按照各子目录里timestamps.txt的第一个时间戳对各个子目录按照升序排列
        self.subsequence_directories = sorted(self.subsequence_directories, key=self.first_time_from_subsequence)

        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(directory.tracks.tracks['t'],
                                                                           directory.images.timestamps)

        len_list = []
        for item in self.img_idx_track_idxs.values():
            num = len(item)
            num = num - (num % batch_size)
            len_list.append(num)
        self.len_list = self.cumsum(len_list)      
        
    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def __len__(self):
        return sum(len(v)-1 for v in self.img_idx_track_idxs.values())

    def __getitem__(self, item):
        output = {}
        img = self.get_image(item)
        events = self.get_events(item)
        events = make_color_histo(events)
        tracks = self.get_tracks(item)
        labels = self.get_labels(item)

        Query_list = []
        Answer_list = []
        for QA in labels["QA"]:
            Q = QA["Query"]
            A = QA["Answer"]
            Query_list.append(Q)
            Answer_list.append(A)

        num = len(Query_list)
        random_num = generate_random_number(num - 1)
        try:
            Query = Query_list[random_num]
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Length of Query_list: {len(Query_list)}")
            print(f"Value of random_num: {random_num}")
        Answer = Answer_list[random_num]

        #process target
        image_id = labels['image_id']
        object_class_list = []
        bbox_list = []
        for t in tracks:
            object_class = t[5]
            xmin = int(t[1])
            ymin = int(t[2])
            xmax = xmin + int(t[3])
            ymax = ymin + int(t[4])
            bbox = [xmin, ymin, xmax, ymax]
            object_class_list.append(object_class)
            bbox_list.append(bbox)

        image_id = int(image_id.replace(".png", ""))
        target = {'image_id': image_id, 'boxes': bbox_list, 'labels': object_class_list}

        #transform
        events = Image.fromarray(events)
        img = Image.fromarray(img)
        target = self.prepare(img, events, target)
        img, events, target = self.transform(img, events, target)

        output['image'] = img
        output['events'] = events
        output['tracks'] = tracks
        output['labels'] = labels

        if self.debug:
            # visualize tracks and events
            events = output['events']
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            output['debug'] = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
            output['debug'] = render_object_detections_on_image(output['debug'], output['tracks'])

            return output
        else:
                       
            dataset_idx = bisect.bisect_right(self.len_list, item)

            samples = {
                "img": img,
                "events": events,
                "target": target,
                "dataset_idx": dataset_idx,
                "Query": Query,
                "Answer": Answer,
                "labels":labels
            }

            return samples

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = index + 1

        return i_0, i_1

    def get_tracks(self, index, mask=None, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        idx0, idx1 = img_idx_to_track_idx[i_1]
        tracks = directory.tracks.tracks[idx0:idx1]

        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        events = extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events

    def get_image(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image_files_path = str(image_files[index])
        image = cv2.imread(image_files_path)
        
        return image
    
    def get_labels(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        label_json = directory.labels.labels[index]
        return label_json


    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
            directory = self.directories[directory_name]
            return index, img_idx_to_track_idx, directory

        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx)-1 <= index:
                index -= (len(img_idx_to_track_idx)-1)
                continue
            else:
                return index, img_idx_to_track_idx, self.directories[f.name]
        else:
            raise ValueError
    
    def cumsum(self, len_list):
        r, acc = [], 0
        for i in range(len(len_list)):
            l = len_list[i]
            r.append(l + acc)
            acc += l
        return r
        
    
def make_color_histo(events, img=None, width=640, height=480):
    """
    simple display function that shows negative events as blue dots and positive as red one
    on a white background
    args :
        - events structured numpy array: timestamp, x, y, polarity.
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int.
        - height int.
    return:
        - img numpy array, height x width x 3.
    """
    if img is None:
        img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 255
    # if events.size:
    assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
    assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

    ON_index = np.where(events['p'] == 1)

    img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['p'][ON_index][:, None]  # red

    OFF_index = np.where(events['p'] == 0)
    img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['p'][OFF_index] + 1)[:,None]  # blue

    return img

def collate_fn(batch):
    # 初始化存储批次数据的字典
    samples = {
        "img": [],
        "events": [],
        "target": [],
        "dataset_idx": [],
        "Query": [],
        "Answer": [],
        "labels":[]
    }

    # 遍历 batch 中的每个样本
    for sample in batch:
        samples["img"].append(sample["img"])
        samples["events"].append(sample["events"])
        samples["target"].append(sample["target"])
        samples["dataset_idx"].append(sample["dataset_idx"])
        samples["Query"].append(sample["Query"])
        samples["Answer"].append(sample["Answer"]),
        samples["labels"].append(sample["labels"])

    # 对需要的键值进行进一步处理（例如转化为 NestedTensor 等）
    samples["img"] = nested_tensor_from_tensor_list(samples["img"])
    samples["events"] = nested_tensor_from_tensor_list(samples["events"])

    return samples
    

class Prepare(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, image, event, target):
        if image:
            w, h = image.size
        else:
            w, h = event.size

        gt = {}
        gt["orig_size"] = torch.as_tensor([int(h), int(w)])
        gt["size"] = torch.as_tensor([int(h), int(w)])

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        if self.dataset == 'dvs':
            boxes = target['boxes']
            classes = target['labels']
        else:
            anno = target["annotations"]
            anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            boxes = [obj["bbox"] for obj in anno]
            classes = [obj["category_id"] for obj in anno]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.dataset == 'coco':
            boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
    
        gt["boxes"] = boxes
        gt["labels"] = classes
        gt["image_id"] = image_id
        
        return gt
    

def generate_random_number(n):
    if n > 0:
        return random.randint(0, n)
    else:
        return 0  
