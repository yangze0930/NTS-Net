import torch.utils.data as data
import os
import sys
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE

def find_classes(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(root, source):

    if not os.path.exists(source):
        print("Setting file %s for haa500_basketball dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                for i in range(duration):
                    item = (clip_path, i+1, target)
                    clips.append(item)
    return clips

def ReadSegmentRGB(path, offsets, frame_index, new_height, new_width, new_length, is_color, name_pattern):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            frame_name = name_pattern % (frame_index)
            frame_path = path + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            if cv_img_origin is None:
               print("Could not load file %s" % (frame_path))
               sys.exit()
               # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input

def ReadSegmentFlow(path, offsets, frame_index, new_height, new_width, new_length, is_color, name_pattern):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            frame_name = name_pattern % (path[-1], frame_index)
            frame_path = path[:-1] + "/" + frame_name
            cv_img_origin = cv2.imread(frame_path, cv_read_flag)
            if cv_img_origin is None:
               print("Could not load file %s" % (frame_path))
               sys.exit()
               # TODO: error handling here
            if new_width > 0 and new_height > 0:
                # use OpenCV3, use OpenCV2.4.13 may have error
                cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
            else:
                cv_img = cv_img_origin
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class haa500_basketball(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d.jpg"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, frame_index, target = self.clips[index]
        average_duration = int(frame_index / self.num_segments)
        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                else:
                    offsets.append(0)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                else:
                    offsets.append(0)
            else:
                print("Only phase train and val are supported.")


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        offsets,
                                        frame_index,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern
                                        )
        elif self.modality == "flow":
            clip_input = ReadSegmentFlow(path,
                                        offsets,
                                        frame_index,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern
                                        )
        else:
            print("No such modality %s" % (self.modality))

        # if self.transform is not None:
        #     clip_input = self.transform(clip_input)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        # if self.video_transform is not None:
        #     clip_input = self.video_transform(clip_input)
        if self.phase == "train":
            if len(clip_input.shape) == 2: # if only 1 img in this batch
                clip_input = np.stack([clip_input] * 3, 2)
            clip_input = Image.fromarray(clip_input, mode='RGB')
            clip_input = transforms.Resize((600, 600), Image.BILINEAR)(clip_input)
            clip_input = transforms.RandomCrop(INPUT_SIZE)(clip_input)
            clip_input = transforms.RandomHorizontalFlip()(clip_input)
            clip_input = transforms.ToTensor()(clip_input)
            clip_input = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(clip_input)

        else:
            if len(clip_input.shape) == 2:
                clip_input = np.stack([clip_input] * 3, 2)
            clip_input = Image.fromarray(clip_input, mode='RGB')
            clip_input = transforms.Resize((600, 600), Image.BILINEAR)(clip_input)
            clip_input = transforms.CenterCrop(INPUT_SIZE)(clip_input)
            clip_input = transforms.ToTensor()(clip_input)
            clip_input = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(clip_input)

        return clip_input, target


    def __len__(self):
        return len(self.clips)
